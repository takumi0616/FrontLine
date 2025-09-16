"""
FrontLine v3 package initializer.

This makes 'src/FrontLine/main_v3' a proper Python package so that
imports like 'from main_v3.main_v3_stage1 import run_stage1' work
from the orchestrator 'src/FrontLine/main_v3.py' and with editor linters (Pylance).
"""

from .main_v3_config import CFG, device, ORIG_H, ORIG_W

# Re-export submodules for convenience (optional; direct submodule imports still recommended)
from . import (
    main_v3_config,
    main_v3_utils,
    main_v3_datasets,
    main_v3_models,
    main_v3_stage1,
    main_v3_stage2_swin,
    main_v3_stage2_diffusion,
    main_v3_stage3,
    main_v3_visualize,
    main_v3_evaluation,
    main_v3_stage4_svg,
    main_v3_shap,
)

__all__ = [
    "CFG", "device", "ORIG_H", "ORIG_W",
    "main_v3_config",
    "main_v3_utils",
    "main_v3_datasets",
    "main_v3_models",
    "main_v3_stage1",
    "main_v3_stage2_swin",
    "main_v3_stage2_diffusion",
    "main_v3_stage3",
    "main_v3_visualize",
    "main_v3_evaluation",
    "main_v3_stage4_svg",
    "main_v3_shap",
]
