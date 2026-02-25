"""
Antibody Design Evaluation Modules

Provides separate modules for scFv/Fab (AntibodyDesignModule) and VHH (NanobodyDesignModule).
"""

from .antibody_module import AntibodyDesignModule
from .nanobody_module import NanobodyDesignModule
from .target_config import load_target_config, get_part1_targets, get_part2_targets, is_part1_target

__all__ = [
    'AntibodyDesignModule', 
    'NanobodyDesignModule',
    'load_target_config',
    'get_part1_targets',
    'get_part2_targets',
    'is_part1_target'
]
