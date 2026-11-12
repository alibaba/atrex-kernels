"""Shape-only bounds for the kernel's signed-int32 workspace indices."""

MAX_PREFILL_TOKENS = 1_048_576
INT32_LIMIT = 1 << 31


def scratch_sizes(rows, br, hkv, splits=1, rpre=1):
    """Return allocation sizes in int32 elements; reject before allocating.

    Keep the inherited four-word union allocation even when compact metadata
    uses fewer words. All products are computed as Python integers here.
    """
    if not (1 <= rows < INT32_LIMIT and 1 <= br <= 16 and hkv >= 1 and splits >= 1):
        raise ValueError("invalid QSA workspace geometry")
    blocks = (rows + br - 1) // br
    entries = br * 513
    tiles = (entries + 15) // 16
    sizes = (blocks * entries * 4 + 128, 3 * blocks,
             blocks * br * 2 * tiles + 4 * max(rpre, 1) + 32)
    # A persistent CTA may fetch one final ticket beyond the last work item.
    if max(*sizes, blocks * hkv * splits + 2048) >= INT32_LIMIT:
        raise ValueError("QSA query/workspace size exceeds signed-int32 indexing capacity")
    return sizes
