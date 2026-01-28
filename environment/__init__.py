"""
Environment Module
==================

World representation and field-of-view management.
"""

from .world import Environment
from .local_view import (
    LocalEnvironment,
    FoVBounds,
    extract_local_bounds,
    create_local_environment,
)

__all__ = [
    'Environment',
    'LocalEnvironment',
    'FoVBounds',
    'extract_local_bounds',
    'create_local_environment',
]
