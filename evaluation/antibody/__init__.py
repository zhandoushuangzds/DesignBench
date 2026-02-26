"""
Antibody Design Evaluation Modules

Provides separate modules for scFv/Fab (AntibodyDesignModule) and VHH (NanobodyDesignModule).
"""

from .antibody_module import AntibodyDesignModule
from .nanobody_module import NanobodyDesignModule
from .target_config import load_target_config, get_part1_targets, get_part2_targets, is_part1_target
from .scaffold_config import (
    load_scaffold_config, 
    get_scaffold_cdr_info, 
    find_scaffold_config_file,
    get_part1_scaffold_info
)

__all__ = [
    'AntibodyDesignModule', 
    'NanobodyDesignModule',
    'load_target_config',
    'get_part1_targets',
    'get_part2_targets',
    'is_part1_target',
    'load_scaffold_config',
    'get_scaffold_cdr_info',
    'find_scaffold_config_file',
    'get_part1_scaffold_info'
]
