"""
FrontLine v4 package initializer.

This makes 'src/FrontLine/main_v4' a proper Python package so that
imports like 'from main_v4.main_v4_stage1 import run_stage1' work
from the orchestrator 'src/FrontLine/main_v4.py' and with editor linters (Pylance).
"""

from .main_v4_config import CFG, device, ORIG_H, ORIG_W

# Re-export submodules for convenience (optional; direct submodule imports still recommended)
from . import (
    main_v4_config,
    main_v4_models,
    main_v4_datasets,
    main_v4_stage1,
    main_v4_stage1_5,
    main_v4_stage2,
    main_v4_stage2_5,
    main_v4_stage3,
    main_v4_stage3_5,
    main_v4_stage4,
    main_v4_stage4_5,
)

__all__ = [
    "CFG", "device", "ORIG_H", "ORIG_W",
    "main_v4_config",
    "main_v4_models",
    "main_v4_datasets",
    "main_v4_stage1",
    "main_v4_stage1_5",
    "main_v4_stage2",
    "main_v4_stage2_5",
    "main_v4_stage3",
    "main_v4_stage3_5",
    "main_v4_stage4",
    "main_v4_stage4_5",
]
