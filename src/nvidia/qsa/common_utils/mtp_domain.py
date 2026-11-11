"""Public-shape eligibility for independent Q1 decode and Q4 verification."""
def eligible(q, k, table):
    batch = table.shape[0]
    # Flattened token/page arithmetic remains signed int32. Actual cache
    # byte addresses are formed by the inherited int64 CuTe pointer helper.
    return (1 <= batch <= 256 and q.shape[0] in (batch, 4 * batch)
            and table.numel() * k.shape[1] < 2**31)
