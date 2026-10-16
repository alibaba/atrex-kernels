"""Short-Q decode first, with capability-based Atrex prefill fallback."""


_PREFILL_READY_DEVICES = set()


def _normalized_call(arguments):
    call = dict(arguments)
    if "page_table" in call:
        call["block_table"] = call["page_table"]
        call["return_softmax_lse"] = call["return_lse"]
        call["s_aux"] = call["learnable_sink"]
        call["window_size"] = (
            -1 if call["window_size_left"] is None else call["window_size_left"],
            -1 if call["window_size_right"] is None else call["window_size_right"],
        )
        call["fa_version"] = call.get("fa_version")
        call["logits_soft_cap"] = None
    return call


def _is_short_q(call):
    query = call.get("max_seqlen_q")
    return query is not None and 1 <= query <= 5


def validate(arguments, *, eligibility_only=False):
    call = _normalized_call(arguments)
    if _is_short_q(call):
        from .decode_runtime import validate as validate_decode

        try:
            validate_decode(call)
            return
        except NotImplementedError:
            pass
    from .prefill_contract import validate as validate_prefill

    validate_prefill(call, eligibility_only=eligibility_only)


def _ensure_prefill_jit(call):
    from .prefill_contract import validate as validate_prefill

    try:
        validate_prefill(call)
    except NotImplementedError:
        return
    if call["q"].shape[0] == 0 or call["max_seqlen_k"] == 0:
        return
    device = call["q"].device
    if device in _PREFILL_READY_DEVICES:
        return
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prewarm the PPU short-Q fallback before CUDA graph capture")
    from .runtime import _launch

    _launch.build(device)
    _PREFILL_READY_DEVICES.add(device)


def forward(*, fa_version=None, **kwargs):
    call = _normalized_call(dict(kwargs, fa_version=fa_version))
    if _is_short_q(call):
        _ensure_prefill_jit(call)
        from .decode_runtime import forward as decode_forward

        try:
            return decode_forward(fa_version=fa_version, **kwargs)
        except NotImplementedError:
            pass
    from .runtime import forward as prefill_forward

    return prefill_forward(**kwargs)
