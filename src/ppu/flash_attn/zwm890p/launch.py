"""Select native decode or prefill using the same side-effect-free contract."""


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


def _select(call, *, eligibility_only=False):
    if _is_short_q(call):
        from .decode_runtime import validate as validate_decode

        try:
            validate_decode(call)
            return "decode"
        except NotImplementedError:
            pass
    from .prefill_contract import validate as validate_prefill

    validate_prefill(call, eligibility_only=eligibility_only)
    return "prefill"


def validate(arguments, *, eligibility_only=False):
    _select(_normalized_call(arguments), eligibility_only=eligibility_only)


def forward(*, fa_version=None, **kwargs):
    call = _normalized_call(dict(kwargs, fa_version=fa_version))
    if _select(call) == "decode":
        from .decode_runtime import forward as decode_forward

        return decode_forward(fa_version=fa_version, **kwargs)
    from .runtime import forward as prefill_forward

    return prefill_forward(**kwargs)
