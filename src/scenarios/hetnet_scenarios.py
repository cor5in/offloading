# src/scenarios/hetnet_scenarios.py
"""
Heterogeneous Network Simulation Scenarios
Implementation of various network deployment and simulation scenarios
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from AIMM_simulator import Sim, MME, Scenario
from ..core.enhanced_cell import EnhancedCell, BSType, BSState
from ..core.traffic_aware_ue import TrafficAwareUE
from ..algorithms.milsf_ric import MiLSF_RIC
from ..utils.logger import MiLSF_Logger
from ..core.hetnet_base import NetworkTopology

def create_hetnet_simulation(config: Dict = None) -> Sim:
    """
    Create a heterogeneous network simulation based on configuration
    
    Args:
        config: Simulation configuration dictionary
        
    Returns:
        Configured AIMM simulation instance
    """
    if config is None:
        from ..config.simulation_config import get_config
        config = get_config('default')
    
    # Create simulation
    sim = Sim(params={
        'fc_GHz': config.get('carrier_frequency', 2.4),
        'h_UT': config.get('ue_height', 2.0),
        'h_BS': config.get('bs_height', 25.0)
    })
    
    # Deploy network based on configuration
    if config.get('deployment_type', 'MHCPP') == 'MHCPP':
        _deploy_mhcpp_network(sim, config)
    else:
        _deploy_ppp_network(sim, config)
    
    return sim

def _deploy_mhcpp_network(sim: Sim, config: Dict):
    """Deploy network using Matérn Hard-Core Point Process"""
    area_size = config.get('area_size', 10000)
    n_macro = config.get('n_macro_cells', 7)
    n_micro = config.get('n_micro_cells', 10)
    n_users = config.get('n_users', 25)
    
    # Deploy macro cells in hexagonal pattern
    if n_macro == 7:  # Standard 7-cell pattern
        center = (area_size/2, area_size/2)
        radius = area_size/4
        macro_positions = NetworkTopology.hexagonal_macro_layout(center, radius, n_rings=1)
    else:
        # Random deployment for other counts
        macro_positions = NetworkTopology.random_micro_layout(
            (area_size*0.2, area_size*0.2, area_size*0.8, area_size*0.8),
            n_macro, min_distance=2000, seed=42
        )
    
    # Deploy macro cells
    for i, (x, y) in enumerate(macro_positions[:n_macro]):
        cell = EnhancedCell(
            sim,
            bs_type=BSType.MACRO,
            xyz=(x, y, 25.0),
            n_subbands=1,
            verbosity=0
        )
    
    # Deploy micro cells with minimum distance constraint
    micro_positions = NetworkTopology.random_micro_layout(
        (area_size*0.1, area_size*0.1, area_size*0.9, area_size*0.9),
        n_micro, min_distance=500, seed=123
    )
    
    for i, (x, y) in enumerate(micro_positions):
        cell = EnhancedCell(
            sim,
            bs_type=BSType.MICRO,
            xyz=(x, y, 10.0),
            n_subbands=1,
            verbosity=0
        )
    
    # Deploy users with hotspots around macro cells
    hotspot_centers = macro_positions[:min(len(macro_positions), 3)]
    user_positions = NetworkTopology.random_ue_layout(
        (area_size*0.1, area_size*0.1, area_size*0.9, area_size*0.9),
        n_users, hotspot_centers=hotspot_centers, hotspot_ratio=0.7, seed=456
    )
    
    for i, (x, y) in enumerate(user_positions):
        ue = TrafficAwareUE(sim, xyz=(x, y, 2.0), verbosity=0)
        # Attach to best cell
        _attach_ue_to_best_cell(ue, sim.cells)

def _deploy_ppp_network(sim: Sim, config: Dict):
    """Deploy network using Poisson Point Process"""
    area_size = config.get('area_size', 10000)
    n_macro = config.get('n_macro_cells', 7)
    n_micro = config.get('n_micro_cells', 10)
    n_users = config.get('n_users', 25)
    
    np.random.seed(42)  # For reproducibility
    
    # Deploy macro cells randomly (PPP)
    for i in range(n_macro):
        x = np.random.uniform(area_size*0.2, area_size*0.8)
        y = np.random.uniform(area_size*0.2, area_size*0.8)
        cell = EnhancedCell(
            sim,
            bs_type=BSType.MACRO,
            xyz=(x, y, 25.0),
            n_subbands=1,
            verbosity=0
        )
    
    # Deploy micro cells randomly (PPP)
    for i in range(n_micro):
        x = np.random.uniform(area_size*0.1, area_size*0.9)
        y = np.random.uniform(area_size*0.1, area_size*0.9)
        cell = EnhancedCell(
            sim,
            bs_type=BSType.MICRO,
            xyz=(x, y, 10.0),
            n_subbands=1,
            verbosity=0
        )
    
    # Deploy users randomly
    for i in range(n_users):
        x = np.random.uniform(area_size*0.1, area_size*0.9)
        y = np.random.uniform(area_size*0.1, area_size*0.9)
        ue = TrafficAwareUE(sim, xyz=(x, y, 2.0), verbosity=0)
        _attach_ue_to_best_cell(ue, sim.cells)

def _attach_ue_to_best_cell(ue: TrafficAwareUE, cells: List[EnhancedCell]):
    """Attach UE to the best available cell"""
    best_cell = None
    best_sinr = -float('inf')
    
    for cell in cells:
        if cell.state == BSState.ACTIVE:
            sinr = cell.calculate_sinr(ue.xyz, cells)
            if sinr > best_sinr and sinr > -6:  # SINR threshold
                best_sinr = sinr
                best_cell = cell
    
    if best_cell:
        ue.attach(best_cell)
        best_cell.attached.add(ue.i)

def create_paper_scenario(scenario_name: str, config: Dict = None) -> Sim:
    """
    Create specific scenarios from the paper
    
    Args:
        scenario_name: Name of the scenario ('scenario_1', 'scenario_2', etc.)
        config: Base configuration to modify
        
    Returns:
        Configured simulation for the specific scenario
    """
    if config is None:
        from ..config.simulation_config import get_config
        config = get_config('default')
    
    if scenario_name == 'scenario_1':
        # PPP vs MHCPP comparison
        return _create_scenario_1(config)
    elif scenario_name == 'scenario_2':
        # User density impact
        return _create_scenario_2(config)
    elif scenario_name == 'scenario_3':
        # SINR threshold impact
        return _create_scenario_3(config)
    elif scenario_name == 'scenario_4':
        # Sleeping cells vs energy savings
        return _create_scenario_4(config)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

def _create_scenario_1(config: Dict) -> Sim:
    """Scenario 1: PPP vs MHCPP deployment comparison"""
    # Use MHCPP by default, PPP variant handled by parameter
    sim = create_hetnet_simulation(config)
    
    # Add MiLSF RIC
    ric = MiLSF_RIC(
        sim,
        interval=30.0,
        low_load_start=22,
        low_load_end=6,
        sinr_threshold=-6.0,
        verbosity=1
    )
    sim.add_ric(ric)
    
    # Add logger
    logger = MiLSF_Logger(sim, logging_interval=60.0)
    sim.add_logger(logger)
    
    return sim

def _create_scenario_2(config: Dict) -> Sim:
    """Scenario 2: User density impact analysis"""
    # Modify user count for density analysis
    user_counts = [10, 25, 50, 75, 100]
    selected_count = config.get('user_count', 25)
    
    config['n_users'] = selected_count
    sim = create_hetnet_simulation(config)
    
    # Add components
    ric = MiLSF_RIC(sim, interval=30.0, verbosity=1)
    sim.add_ric(ric)
    
    logger = MiLSF_Logger(sim, logging_interval=60.0, 
                         filename_prefix=f"scenario2_users_{selected_count}")
    sim.add_logger(logger)
    
    return sim

def _create_scenario_3(config: Dict) -> Sim:
    """Scenario 3: SINR threshold impact analysis"""
    sinr_thresholds = [-12, -9, -6, -3, 0]
    selected_threshold = config.get('sinr_threshold', -6)
    
    sim = create_hetnet_simulation(config)
    
    # Add MiLSF with specific SINR threshold
    ric = MiLSF_RIC(
        sim,
        interval=30.0,
        sinr_threshold=selected_threshold,
        verbosity=1
    )
    sim.add_ric(ric)
    
    logger = MiLSF_Logger(sim, logging_interval=60.0,
                         filename_prefix=f"scenario3_sinr_{abs(selected_threshold)}")
    sim.add_logger(logger)
    
    return sim

def _create_scenario_4(config: Dict) -> Sim:
    """Scenario 4: Sleeping cells vs energy savings relationship"""
    sim = create_hetnet_simulation(config)
    
    # Add detailed MiLSF for sleep tracking
    ric = MiLSF_RIC(sim, interval=30.0, verbosity=2)
    sim.add_ric(ric)
    
    # Use detailed logger for comprehensive sleep tracking
    from ..utils.logger import DetailedLogger
    logger = DetailedLogger(sim, logging_interval=30.0,
                           filename_prefix="scenario4_sleep_analysis")
    sim.add_logger(logger)
    
    return sim

def create_custom_scenario(network_config: Dict, algorithm_config: Dict = None) -> Sim:
    """
    Create a custom simulation scenario
    
    Args:
        network_config: Network topology configuration
        algorithm_config: Algorithm parameters configuration
        
    Returns:
        Configured simulation
    """
    sim = create_hetnet_simulation(network_config)
    
    if algorithm_config is None:
        algorithm_config = {
            'interval': 30.0,
            'low_load_start': 22,
            'low_load_end': 6,
            'sinr_threshold': -6.0,
            'verbosity': 1
        }
    
    # Add RIC with custom configuration
    ric = MiLSF_RIC(sim, **algorithm_config)
    sim.add_ric(ric)
    
    # Add MME for handover management
    mme = MME(sim, interval=10.0, verbosity=0)
    sim.add_mme(mme)
    
    # Add logger
    logger = MiLSF_Logger(sim, logging_interval=60.0,
                         filename_prefix="custom_scenario")
    sim.add_logger(logger)
    
    return sim

class TrafficPatternScenario(Scenario):
    """Dynamic traffic pattern scenario"""
    
    def __init__(self, sim, traffic_multiplier=1.0, **kwargs):
        super().__init__(sim, **kwargs)
        self.traffic_multiplier = traffic_multiplier
        
    def loop(self):
        """Update traffic patterns over time"""
        while True:
            current_hour = int(self.sim.env.now / 3600) % 24
            
            # Update all UE traffic patterns
            for ue in self.sim.UEs:
                if hasattr(ue, 'update_traffic'):
                    ue.update_traffic()
                    
            # Adjust global traffic based on time
            if 6 <= current_hour <= 22:  # Active hours
                self.traffic_multiplier = 1.0
            else:  # Low-load period
                self.traffic_multiplier = 0.3
                
            yield self.sim.wait(3600)  # Update every hour

def create_dynamic_traffic_scenario(config: Dict = None) -> Sim:
    """Create scenario with dynamic traffic patterns"""
    if config is None:
        from ..config.simulation_config import get_config
        config = get_config('default')
        
    sim = create_hetnet_simulation(config)
    
    # Add dynamic traffic scenario
    traffic_scenario = TrafficPatternScenario(sim, interval=3600)
    sim.add_scenario(traffic_scenario)
    
    # Add MiLSF RIC
    ric = MiLSF_RIC(sim, interval=30.0, verbosity=1)
    sim.add_ric(ric)
    
    # Add logger
    logger = MiLSF_Logger(sim, logging_interval=300.0,
                         filename_prefix="dynamic_traffic")
    sim.add_logger(logger)
    
    return sim

def run_comparative_study(scenarios: List[str], base_config: Dict = None):
    """
    Run comparative study across multiple scenarios
    
    Args:
        scenarios: List of scenario names to compare
        base_config: Base configuration for all scenarios
    """
    if base_config is None:
        from ..config.simulation_config import get_config
        base_config = get_config('default')
    
    results = {}
    
    for scenario_name in scenarios:
        print(f"\n=== Running {scenario_name} ===")
        
        try:
            sim = create_paper_scenario(scenario_name, base_config.copy())
            sim.run(until=base_config.get('duration', 86400))
            
            # Collect results
            if hasattr(sim, 'ric') and sim.ric:
                energy_savings = sim.ric.total_energy_without_sleeping
                if energy_savings > 0:
                    savings_pct = ((energy_savings - sim.ric.total_energy_with_sleeping) 
                                  / energy_savings) * 100
                    results[scenario_name] = {
                        'energy_savings': savings_pct,
                        'sleep_decisions': sim.ric.sleep_decisions,
                        'wake_decisions': sim.ric.wake_decisions,
                        'failed_reallocations': sim.ric.failed_reallocations
                    }
                    
        except Exception as e:
            print(f"Error running {scenario_name}: {e}")
            results[scenario_name] = {'error': str(e)}
    
    # Print comparison results
    print("\n=== Comparative Results ===")
    for scenario, result in results.items():
        if 'error' not in result:
            print(f"{scenario}:")
            print(f"  Energy Savings: {result['energy_savings']:.2f}%")
            print(f"  Sleep Decisions: {result['sleep_decisions']}")
            print(f"  Failed Reallocations: {result['failed_reallocations']}")
        else:
            print(f"{scenario}: ERROR - {result['error']}")
    
    return results

# Export main functions
__all__ = [
    'create_hetnet_simulation',
    'create_paper_scenario',
    'create_custom_scenario',
    'create_dynamic_traffic_scenario',
    'run_comparative_study',
    'TrafficPatternScenario'
]

if __name__ == "__main__":
    # Test scenario creation
    print("Testing scenario creation...")
    
    from ..config.simulation_config import get_config
    config = get_config('small_network')
    
    # Test basic network creation
    sim = create_hetnet_simulation(config)
    print(f"Created network with {len(sim.cells)} cells and {len(sim.UEs)} UEs")
    
    # Test paper scenario
    sim2 = create_paper_scenario('scenario_1', config)
    print(f"Created scenario 1 with RIC: {hasattr(sim2, 'ric')}")
    
    print("Scenario testing completed!")