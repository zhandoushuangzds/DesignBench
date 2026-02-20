"""
Antibody Design Evaluation Modules

Provides separate modules for scFv/Fab (AntibodyDesignModule) and VHH (NanobodyDesignModule).
"""

from .antibody_module import AntibodyDesignModule
from .nanobody_module import NanobodyDesignModule

__all__ = ['AntibodyDesignModule', 'NanobodyDesignModule']
