"""Static Q1/Q4 choices from full-bucket and serving-layout screens.

Requests are allocated tensor/graph slots, including padding. Selection never
reads token contents or synchronizes the device. Unmeasured shapes and layouts
keep the existing architecture-specific configuration.
"""

from .decode_intervals import _select_fp8_q4


def _standard_layout(q, k, v, ids, pos, table, req, gate, out):
    return (q.shape[-1] == k.shape[-1] == v.shape[-1] == 256
            and all(t.is_contiguous() for t in (q, k, v, ids, pos, table, req, gate, out))
            and all(t.data_ptr() % 32 == 0 for t in (q, k, v, gate, out))
            and k.stride(0) % 64 == 0 and k.stride(1) % 64 == 0)


def _serving_layout(q, k, v, ids, pos, table, req, gate, out, *, layers=18):
    """Aligned 18-layer serving strides; selection does not depend on aliasing.

    Every selected property is already part of the static launch signature.
    K and V retain independent pointers, including when they share a pool.
    """
    if k.shape != v.shape or k.shape[-2:] != (1, 256) or q.shape[-1] != 256:
        return False
    expected = (layers * k.shape[1] * 512, 512, 512, 1)
    return (k.stride() == v.stride() == expected
            and all(t.is_contiguous() for t in (q, ids, pos, table, req, gate, out))
            and all(t.data_ptr() % 32 == 0 for t in (q, k, v, gate, out)))


def _select_decode_config(rows, requests, hkv, group, selected_blocks, sms,
                          fp8, *, sm100, page, standard_layout, serving_layout=False,
                          single_layer_layout=False, kv_capacity=0):
    if (group == 16 and hkv == 1 and sms in (148, 152)
            and selected_blocks == 512 and 1 <= requests <= 128
            and page in (128, 1056, 2112)
            and (standard_layout or serving_layout or single_layer_layout)):
        if rows == requests:
            if fp8 and sm100 and 64 <= requests <= 72 and 0 < kv_capacity <= 2112:
                return 1, 1
        elif rows == 4 * requests:
            if fp8 and 32 <= requests <= 40:
                return (1, 1) if 4 * requests <= sms else (2, 1)
            if not fp8 and page == 128 and 48 <= requests <= sms // 2:
                return 4, (3 if 3 * requests <= sms else 2)
            if not fp8 and 48 <= requests <= 64:
                return 2, 1
            if requests > sms // 2:
                return 4, 1
        return None
    if ((serving_layout or (standard_layout and sms == 148))
            and sms in (148, 152) and selected_blocks == 512
            and group == 8 and hkv == 1 and rows == 4 * requests
            and page in (128, 1056, 2112)):
        # These layouts/SM counts have their own measured scope. Do not apply
        # the older 152-SM contiguous interval table to arbitrary strided KV.
        if fp8 and serving_layout and 32 <= requests <= sms // 4:
            return (2 if sm100 else 1), 2
        if (fp8 and serving_layout and sm100 and sms == 148
                and page == 128 and requests == 128):
            return 2, 1
        # On 152-SM SM100, the 80-slot partial batch loses warm-replay time.
        # Retain its default route; the measured improvement starts at 96.
        if fp8 and (96 if sm100 and sms == 152 else 80) <= requests <= 128:
            return 4, 1
        if not fp8 and 40 <= requests <= 64:
            return 2, 1
    if (not standard_layout or sms != 152 or selected_blocks != 512
            or group not in (8, 12) or hkv not in (1, 2)
            or page not in (128, 256, 512, 1024, 1056, 2112)
            or not 1 <= requests <= 256):
        return None
    # Pair queries to reuse KV while retaining enough independent CTAs.
    if (not fp8 and page in (128, 256, 1056, 2112) and rows == 4 * requests
            and ((hkv == 1 and 40 <= requests <= 64)
                 or (hkv == 2 and 20 <= requests <= 24))):
        return 2, 1
    # A single Q4 CTA avoids split buffers; runtimes select three pipeline stages.
    if (fp8 and page in (128, 256, 1056, 2112) and rows == 4 * requests
            and hkv == 1 and 80 <= requests <= 128):
        return 4, 1
    if page == 2112:
        return None
    if rows == requests:
        units = requests * hkv
        if not fp8 and 152 < units <= 168:
            return 1, 4
        if not fp8 and 168 < units <= 176:
            return 1, 2
        if fp8 and hkv == 1 and group == 12 and 64 <= requests <= 72:
            return 1, 1
    elif rows == 4 * requests:
        if fp8:
            config = _select_fp8_q4(requests, hkv, group, sm100, page)
            if config is not None:
                return config
        if fp8 and hkv == 2 and 20 <= requests <= 38:
            return 2, 1
        if (hkv == 2 and 40 <= requests < 64
                and (not fp8 or sm100 or requests >= 58)):
            return 4, 1
        if fp8 and hkv == 1:
            if 32 <= requests <= 38:
                return 1, 1
            if 156 <= requests <= 224:
                return 2, 1
        if not fp8 and hkv == 1:
            if sm100 and group == 8:
                if 40 <= requests < 64 or requests == 76 or 160 <= requests <= 228:
                    return 2, 1
            if requests == 76:
                return 1, 1
    return None
