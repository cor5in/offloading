# src/scenarios/__init__.py
"""
Predefined simulation scenarios
"""

from .hetnet_scenarios import (
    create_hetnet_simulation,
    create_paper_scenario,
    create_custom_scenario
)

__all__ = [
    'create_hetnet_simulation',
    'create_paper_scenario', 
    'create_custom_scenario'
]