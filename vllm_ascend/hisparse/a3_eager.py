# SPDX-License-Identifier: Apache-2.0
"""A3 eager compatibility path without SIMT or AIV-mapped Host memory.

Planning synchronizes metadata to CPU. This path is for functional bring-up,
not graph capture or performance qualification. KV payload stays in the pinned
Host pool until selected rows are copied to the NPU.
"""

import torch


def enabled():
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType

    return get_ascend_device_type() == AscendDeviceType.A3


def require_eager():
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("A3 HiSparse compatibility path requires --enforce-eager")


class PinnedHostTensor:
    def __init__(self, tensor):
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or not tensor.is_pinned():
            raise ValueError("A3 HiSparse requires contiguous pinned CPU storage")
        self.tensor = tensor
        self.device = torch.device("npu", torch.npu.current_device())
        self.closed = False

    def close(self):
        self.closed = True


def _resident_only(a, cpu):
    """Vectorize the common Resident-only case without reading Hot/LRU state."""
    resident, source = cpu["resident_block_table"], cpu["source_block_table"]
    if resident is None or source is None:
        return None
    selected = cpu["global_indices"]
    rows, topk = selected.shape
    active = rows if cpu["num_valid_rows"] is None else max(0, min(rows, int(cpu["num_valid_rows"].item())))
    requests, states = cpu["request_ids"], cpu["request_state_indices"]
    out = torch.full_like(selected, -1)
    resolved = out.clone()
    for row in range(active):
        request = row if requests is None else int(requests[row])
        if request < 0 or (states is not None and request >= states.numel()):
            continue
        state = request if states is None else int(states[request])
        if not 0 <= state < a["device_global_indices"].shape[0] or request >= a["hot_block_table"].shape[0]:
            continue
        if state >= resident.shape[0] or request >= source.shape[0] or min(source.shape[1], resident.shape[1]) == 0:
            return None
        token = selected[row]
        sp = token.clamp_min(0).div(cpu["source_block_size"], rounding_mode="floor").long()
        rp = token.clamp_min(0).div(cpu["resident_block_size"], rounding_mode="floor").long()
        valid = (token >= 0) & (sp < source.shape[1])
        host_page = source[request, sp.clamp_max(source.shape[1] - 1)]
        host = host_page * cpu["source_block_size"] + token.remainder(cpu["source_block_size"])
        host_valid = valid & (host_page > 0) & (host >= 0) & (host < a["host_cache"].shape[0])
        resident_page = resident[state, rp.clamp_max(resident.shape[1] - 1)]
        resident_valid = valid & (rp < resident.shape[1]) & (resident_page >= 0)
        resident_valid &= resident_page != cpu["resident_null_block"]
        if torch.any(host_valid & ~resident_valid):
            return None
        resolved[row] = torch.where(host_valid, host, -1)
        physical = resident_page * cpu["resident_block_size"] + token.remainder(cpu["resident_block_size"])
        out[row] = torch.where(resident_valid, physical, -1)
    block = a["hot_cache"].shape[1]
    return {
        "hot_indices": out,
        "resolved_global_indices": resolved,
        "miss_mask": torch.zeros_like(out),
        "valid_counts": (out >= 0).sum(-1).to(torch.int32),
        "swap_host_physical_rows": torch.full_like(out, -1),
        "swap_device_physical_rows": torch.full_like(out, -1),
        "swap_counts": torch.zeros(rows, dtype=torch.int32),
        "attention_indices": torch.where(
            out >= 0, out.div(block, rounding_mode="floor") * cpu["attention_block_stride"] + out.remainder(block), -1
        ),
        "device_global_indices": a["device_global_indices"],
        "lru_slots": a["lru_slots"],
        "stats": torch.zeros(2, dtype=torch.int64),
    }


def resolve_cpu(a):
    """Resolve a request union before modifying any eviction candidates."""
    cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in a.items()
        if k
        in (
            "global_indices",
            "request_ids",
            "request_state_indices",
            "source_block_table",
            "resident_block_table",
            "num_valid_rows",
            "resident_null_block",
            "source_block_size",
            "resident_block_size",
            "max_union_rows",
            "attention_block_stride",
        )
    }
    fast = _resident_only(a, cpu)
    if fast is not None:
        return fast
    for key in ("device_global_indices", "lru_slots", "hot_block_table"):
        cpu[key] = a[key].cpu()
    selected = cpu["global_indices"]
    rows, topk = selected.shape
    active = rows if cpu["num_valid_rows"] is None else max(0, min(rows, int(cpu["num_valid_rows"].item())))
    requests = cpu["request_ids"]
    requests = list(range(rows)) if requests is None else requests.tolist()
    states = cpu["request_state_indices"]
    dgi, lru = cpu["device_global_indices"].clone(), cpu["lru_slots"].clone()
    out = torch.full((rows, topk), -1, dtype=torch.int32)
    resolved = out.clone()
    miss = torch.zeros_like(out)
    counts = torch.zeros(rows, dtype=torch.int32)
    sh, sd = out.clone(), out.clone()
    sc = counts.clone()
    stats = torch.zeros(2, dtype=torch.int64)
    block = a["hot_cache"].shape[1]
    host_rows = a["host_cache"].shape[0]
    source, resident, table = cpu["source_block_table"], cpu["resident_block_table"], cpu["hot_block_table"]
    groups = {}
    for row in range(active):
        request = requests[row]
        if request >= 0:
            groups.setdefault(request, []).append(row)
    for request, query_rows in groups.items():
        if states is not None and request >= states.numel():
            continue
        state = request if states is None else int(states[request])
        if not 0 <= state < dgi.shape[0] or request >= table.shape[0]:
            continue
        union, refs = {}, []
        for row in query_rows:
            for col, token in enumerate(selected[row].tolist()):
                if token < 0:
                    continue
                host = token
                if source is not None:
                    page, offset = divmod(token, cpu["source_block_size"])
                    if request >= source.shape[0] or page >= source.shape[1]:
                        continue
                    source_page = int(source[request, page])
                    host = source_page * cpu["source_block_size"] + offset if source_page > 0 else -1
                physical = -1
                if resident is not None:
                    page, offset = divmod(token, cpu["resident_block_size"])
                    if state < resident.shape[0] and page < resident.shape[1]:
                        resident_page = int(resident[state, page])
                        if resident_page >= 0 and resident_page != cpu["resident_null_block"]:
                            physical = resident_page * cpu["resident_block_size"] + offset
                host = host if 0 <= host < host_rows else -1
                resolved[row, col] = host
                if physical >= 0:
                    out[row, col] = physical
                elif host >= 0:
                    union.setdefault(host, None)
                    refs.append((row, col, host))
                else:
                    continue
                counts[row] += 1
        if len(union) > cpu["max_union_rows"]:
            raise ValueError("A3 HiSparse selected union exceeds prepared capacity")
        order = lru[state].tolist()
        hits = [slot for slot in order if int(dgi[state, slot]) in union]
        hit_set = set(hits)
        victims = [slot for slot in order if slot not in hit_set]
        mapping = {int(dgi[state, slot]): slot for slot in hits}
        missing = []
        for host in union:
            if host not in mapping:
                if len(missing) >= len(victims):
                    raise ValueError("A3 HiSparse Hot cache cannot hold the selected union")
                slot = victims[len(missing)]
                mapping[host] = slot
                dgi[state, slot] = host
                missing.append((host, slot))
        if union:
            lru[state] = torch.tensor(victims[len(missing) :] + victims[: len(missing)] + hits, dtype=lru.dtype)
        first_miss = set(host for host, _ in missing)
        for row, col, host in refs:
            slot = mapping[host]
            out[row, col] = int(table[request, slot // block]) * block + slot % block
            if host in first_miss:
                miss[row, col] = 1
                first_miss.remove(host)
        for i, (host, slot) in enumerate(missing):
            row, col = query_rows[i // topk], i % topk
            sh[row, col] = host
            sd[row, col] = int(table[request, slot // block]) * block + slot % block
            sc[row] += 1
        stats += torch.tensor([len(refs) - len(missing), len(missing)])
    attention = torch.where(
        out >= 0, out.div(block, rounding_mode="floor") * cpu["attention_block_stride"] + out.remainder(block), -1
    )
    return {
        "hot_indices": out,
        "resolved_global_indices": resolved,
        "miss_mask": miss,
        "valid_counts": counts,
        "swap_host_physical_rows": sh,
        "swap_device_physical_rows": sd,
        "swap_counts": sc,
        "device_global_indices": dgi,
        "lru_slots": lru,
        "attention_indices": attention,
        "stats": stats,
    }


def resolve(owner, arguments):
    require_eager()
    if owner.attention_capacity:
        raise ValueError("A3 eager HiSparse currently supports BF16 KV only")
    result = resolve_cpu(arguments)
    if result["stats"][1] > 0:
        from vllm.logger import logger

        logger.info_once("A3 HiSparse eager Host refill activated.")
    for key, value in result.items():
        destination = arguments.get(key)
        if destination is not None:
            if destination is value:
                continue
            if key == "stats":
                if torch.any(value):
                    destination.add_(value.to(destination.device))
            else:
                destination.copy_(value.reshape(destination.shape))


def gather(host, hot, source, destination, counts, mask, state, attention, attention_block):
    require_eager()
    counts_cpu = None if counts is None else counts.cpu()
    if counts_cpu is not None and not torch.any(counts_cpu) and attention is None:
        return
    src, dst = source.cpu(), destination.cpu()
    selected = (
        torch.arange(src.shape[1])[None, :] < counts_cpu[:, None] if counts_cpu is not None else mask.cpu().bool()
    )
    selected &= (src >= 0) & (src < host.shape[0]) & (dst >= 0) & (dst < hot.shape[0] * hot.shape[1])
    if state is not None:
        selected &= state.cpu()[:, None] >= 0
    source_ids, target_ids = src[selected].long(), dst[selected].long()
    if source_ids.numel():
        payload = host.index_select(0, source_ids).to(hot.device)
        # Index both dimensions to preserve gaps between shared-slab pages.
        hot[
            target_ids.div(hot.shape[1], rounding_mode="floor").to(hot.device),
            target_ids.remainder(hot.shape[1]).to(hot.device),
        ] = payload
    if attention is not None:
        mapped = dst.div(hot.shape[1], rounding_mode="floor") * attention_block + dst.remainder(hot.shape[1])
        attention.copy_(torch.where(dst >= 0, mapped, -1))
