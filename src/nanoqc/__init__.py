"""Quantum-classical VHH-antigen interface side-chain reconstruction benchmark."""

import warnings


# torch-geometric 2.8 scripts SelectOutput while importing its pool module.
# PyTorch 2.14 emits this warning from torch.jit._script for that dependency
# call; it is not a call made by nanoqc. Keep every other FutureWarning visible.
warnings.filterwarnings(
    "ignore",
    message=r"^`torch\.jit\.script` is deprecated\. Please switch to `torch\.compile` or `torch\.export`\.$",
    category=FutureWarning,
    module=r"^torch\.jit\._script$",
)
