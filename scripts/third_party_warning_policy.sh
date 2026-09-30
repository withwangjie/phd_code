#!/usr/bin/env bash
# PyG 2.8 imports torch_geometric.nn.pool.select.base, which calls the
# deprecated torch.jit.script API. Install this exact warning filter before
# Python starts so it also covers workers that import PyG before nanoqc.
# PYTHONWARNINGS entries are evaluated from left to right; the last match wins.
_PYG_TORCHSCRIPT_WARNING='ignore:`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.:FutureWarning:torch.jit._script'
case ",${PYTHONWARNINGS:-}," in
    *",${_PYG_TORCHSCRIPT_WARNING},"*) ;;
    *) export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${_PYG_TORCHSCRIPT_WARNING}" ;;
esac
unset _PYG_TORCHSCRIPT_WARNING
