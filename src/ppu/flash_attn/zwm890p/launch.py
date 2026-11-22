"""Select native decode or prefill using the same side-effect-free contract."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _decode_runtime():
    from . import decode_runtime

    return decode_runtime


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
        call["logits_soft_cap"] = None
    return call


def _is_short_q(call):
    query = call.get("max_seqlen_q")
    return query is not None and 1 <= query <= 5


def _select(call, *, common_validated=False):
    if _is_short_q(call):
        try:
            _decode_runtime().validate(call, common_validated=common_validated)
            return "decode"
        except NotImplementedError:
            pass
    from .prefill_contract import validate as validate_prefill

    validate_prefill(call)
    return "prefill"


def validate(arguments):
    _select(_normalized_call(arguments))


def forward(*, fa_version=None, **kwargs):
    call = _normalized_call(dict(kwargs, fa_version=fa_version))
    if _select(call) == "decode":
        return _decode_runtime().forward(fa_version=fa_version, **kwargs)
    from .prefill_runtime import forward as prefill_forward

    return prefill_forward(**kwargs)
