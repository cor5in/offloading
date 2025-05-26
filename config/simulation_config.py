# config/simulation_config.py
"""
Configuration parameters for MiLSF simulation
Based on Table II from the paper
"""

# Network Configuration
NETWORK_CONFIG = {
    # Simulation area
    'area_size': 10000,  # 10km x 10km (meters)
    
    # Cell deployment
    'n_macro_cells': 7,   # Hexagonal deployment
    'n_micro_cells': 10,  # Random deployment
    'n_users': 25,        # User density as in paper
    
    # Neighbor threshold
    'neighbor_threshold': 3000,  # 3km for neighbor cell detection
}

# Macro Base Station Parameters (Table II)
MACRO_BS_CONFIG = {
    'transmit_power_dBm': 46.0,      # 8W per antenna (logarithmic)
    'transmit_power_linear': 8.0,     # 8W per antenna (linear)
    'n_antennas': 6,                  # α₁ = 6
    'carrier_frequency': 2.4e9,       # f_c1 = 2.4 GHz
    'bandwidth': 20e6,                # w₁ = 20 MHz
    'circuit_power': 120.0,           # p_c1 = 120 W
    'sleep_power': 8.0,               # p_s1 = 8 W
    'antenna_height': 25.0,           # h_BS = 25 m
    'coverage_radius': 2700.0,        # Approximate coverage
}

# Micro Base Station Parameters (Table II)
MICRO_BS_CONFIG = {
    'transmit_power_dBm': 38.0,       # 3W per antenna (logarithmic)
    'transmit_power_linear': 3.0,     # 3W per antenna (linear)
    'n_antennas': 2,                  # α₂ = 2
    'carrier_frequency': 20e9,        # f_c2 = 20 GHz
    'bandwidth': 50e6,                # w₂ = 50 MHz
    'circuit_power': 10.0,            # p_c2 = 10 W
    'sleep_power': 2.0,               # p_s2 = 2 W
    'antenna_height': 10.0,           # Typical micro BS height
    'coverage_radius': 1000.0,        # Approximate coverage
}

# UE Parameters (Table II)
UE_CONFIG = {
    'antenna_height': 2.0,            # h_UT = 2 m
    'mobility': 'stationary',         # Stationary during low-load period
    'noise_power_dBm': -100.0,        # η₀ = -100 dBm
    'base_traffic_rate': 1.0,         # Base traffic in Mbps
    'traffic_variation': 0.5,         # Traffic variation factor
}

# Path Loss Model Parameters
PATHLOSS_CONFIG = {
    'model_type': '3GPP_UMa',         # 3GPP Urban Macro model
    'pathloss_exponent': 3.7,         # β = 3.7
    'speed_of_light': 3e8,            # c = 3×10⁸ m/s
    'min_distance': 10.0,             # Minimum distance for path loss
}

# MiLSF Algorithm Parameters
MILSF_CONFIG = {
    'sinr_threshold_dB': -6.0,        # γ₀ = -6 dB
    'low_load_start_hour': 22,        # 10:00 PM
    'low_load_end_hour': 6,           # 6:00 AM
    'ric_interval': 30.0,             # RIC decision interval (seconds)
    'load_threshold': 1.0,            # Maximum cell load
    'reallocation_strategy': 'milsf', # User reallocation strategy
}

# Traffic Prediction Parameters
TRAFFIC_PREDICTION_CONFIG = {
    'prediction_model': 'BLSTM',      # Bidirectional LSTM
    'sequence_length': 24,            # 24 hours input sequence
    'hidden_units': 500,              # LSTM hidden units (paper specification)
    'dropout_rate': 0.2,              # Dropout for regularization
    'learning_rate': 0.001,           # RMSprop learning rate
    'batch_size': 32,                 # Training batch size
    'epochs': 50,                     # Training epochs
    'validation_split': 0.2,          # Validation data ratio
}

# Simulation Parameters
SIMULATION_CONFIG = {
    'duration': 86400,                # 24 hours in seconds
    'time_step': 1.0,                 # 1 second time step
    'logging_interval': 60.0,         # Log every minute
    'random_seed': 42,                # For reproducible results
    'verbosity': 1,                   # Output verbosity level
}

# Performance Metrics
METRICS_CONFIG = {
    'energy_calculation_interval': 30.0,  # Energy calculation frequency
    'qos_violation_threshold': -6.0,      # SINR below this is QoS violation
    'load_violation_threshold': 1.0,      # Load above this is violation
    'save_detailed_logs': True,           # Save detailed performance logs
    'save_energy_history': True,          # Save energy history
}

# Deployment Scenarios (for different experiments)
DEPLOYMENT_SCENARIOS = {
    'paper_reproduction': {
        'area_size': 10000,
        'n_macro_cells': 7,
        'n_micro_cells': 10,
        'n_users': 25,
        'deployment_type': 'MHCPP',  # Matérn Hard-Core Point Process
    },
    
    'small_network': {
        'area_size': 5000,
        'n_macro_cells': 3,
        'n_micro_cells': 5,
        'n_users': 15,
        'deployment_type': 'MHCPP',
    },
    
    'large_network': {
        'area_size': 15000,
        'n_macro_cells': 19,  # Larger hexagonal pattern
        'n_micro_cells': 30,
        'n_users': 100,
        'deployment_type': 'MHCPP',
    },
    
    'dense_network': {
        'area_size': 10000,
        'n_macro_cells': 7,
        'n_micro_cells': 20,
        'n_users': 50,
        'deployment_type': 'MHCPP',
    },
}

# Experimental Scenarios from Paper
PAPER_SCENARIOS = {
    'scenario_1': {
        'description': 'PPP vs MHCPP deployment comparison',
        'deployment_types': ['PPP', 'MHCPP'],
        'metrics': ['energy_savings', 'qos_violations'],
    },
    
    'scenario_2': {
        'description': 'User density impact',
        'user_counts': [10, 25, 50, 75, 100],
        'metrics': ['energy_savings', 'sleeping_cells'],
    },
    
    'scenario_3': {
        'description': 'SINR threshold impact', 
        'sinr_thresholds': [-12, -9, -6, -3, 0],
        'metrics': ['energy_savings', 'user_satisfaction'],
    },
    
    'scenario_4': {
        'description': 'Sleeping cells vs energy savings',
        'metrics': ['sleeping_cells_count', 'energy_savings_percentage'],
    },
}

# Default configuration combining all parameters
DEFAULT_CONFIG = {
    **NETWORK_CONFIG,
    'macro_bs': MACRO_BS_CONFIG,
    'micro_bs': MICRO_BS_CONFIG,
    'ue': UE_CONFIG,
    'pathloss': PATHLOSS_CONFIG,
    'milsf': MILSF_CONFIG,
    'traffic_prediction': TRAFFIC_PREDICTION_CONFIG,
    'simulation': SIMULATION_CONFIG,
    'metrics': METRICS_CONFIG,
}

def get_config(scenario_name='default'):
    """Get configuration for specific scenario"""
    
    if scenario_name == 'default':
        return DEFAULT_CONFIG.copy()
    elif scenario_name in DEPLOYMENT_SCENARIOS:
        config = DEFAULT_CONFIG.copy()
        config.update(DEPLOYMENT_SCENARIOS[scenario_name])
        return config
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

def validate_config(config):
    """Validate configuration parameters"""
    
    errors = []
    
    # Check required parameters
    required_params = [
        'area_size', 'n_macro_cells', 'n_micro_cells', 'n_users'
    ]
    
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
            
    # Check parameter ranges
    if config.get('area_size', 0) <= 0:
        errors.append("area_size must be positive")
        
    if config.get('n_macro_cells', 0) <= 0:
        errors.append("n_macro_cells must be positive")
        
    if config.get('n_users', 0) <= 0:
        errors.append("n_users must be positive")
        
    # Check SINR threshold
    sinr_threshold = config.get('milsf', {}).get('sinr_threshold_dB', -6)
    if sinr_threshold > 10 or sinr_threshold < -20:
        errors.append("SINR threshold should be between -20 and 10 dB")
        
    # Check time periods
    low_load_start = config.get('milsf', {}).get('low_load_start_hour', 22)
    low_load_end = config.get('milsf', {}).get('low_load_end_hour', 6)
    
    if not (0 <= low_load_start <= 23):
        errors.append("low_load_start_hour must be between 0 and 23")
        
    if not (0 <= low_load_end <= 23):
        errors.append("low_load_end_hour must be between 0 and 23")
        
    return errors

def print_config(config):
    """Print configuration in readable format"""
    
    print("=== Simulation Configuration ===")
    print(f"Network: {config['area_size']/1000:.1f}km × {config['area_size']/1000:.1f}km")
    print(f"Cells: {config['n_macro_cells']} macro + {config['n_micro_cells']} micro")
    print(f"Users: {config['n_users']}")
    
    if 'milsf' in config:
        milsf_cfg = config['milsf']
        print(f"SINR Threshold: {milsf_cfg['sinr_threshold_dB']} dB")
        print(f"Low-load Period: {milsf_cfg['low_load_start_hour']}:00 - {milsf_cfg['low_load_end_hour']}:00")
        
    if 'simulation' in config:
        sim_cfg = config['simulation']
        print(f"Duration: {sim_cfg['duration']/3600:.1f} hours")
        print(f"Logging Interval: {sim_cfg['logging_interval']} seconds")
        
    print("=" * 32)

if __name__ == "__main__":
    # Test configuration
    config = get_config('default')
    print_config(config)
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
        
    # Test different scenarios
    for scenario in ['small_network', 'large_network', 'dense_network']:
        print(f"\n--- {scenario} ---")
        scenario_config = get_config(scenario)
        print_config(scenario_config)