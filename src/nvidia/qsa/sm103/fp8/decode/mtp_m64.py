"""SM103 M64 launch and softmax with shared decode roles."""
from ....common_utils.fp8.decode.mtp_m64 import QSAForward as M64
from .qsa_kernel import QSAForward as SM103


class QSAForward(M64, SM103):
    """Use SM103 entry/softmax and architecture-neutral M64 roles."""
