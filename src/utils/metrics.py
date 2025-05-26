# src/utils/metrics.py
"""
Performance metrics calculation for MiLSF simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from math import log2
from ..core.enhanced_cell import BSState, BSType

class EnergyMetrics:
    """Energy efficiency metrics calculator"""
    
    def __init__(self):
        self.energy_history = deque(maxlen=1000)
        self.baseline_energy_history = deque(maxlen=1000)
        
    def calculate_instantaneous_energy_savings(self, cells) -> float:
        """Calculate instantaneous energy savings percentage"""
        total_current_power = 0
        total_baseline_power = 0
        
        for cell in cells:
            if hasattr(cell, 'get_power_consumption'):
                current_power = cell.get_power_consumption()
                total_current_power += current_power
                
                # Calculate baseline power (all cells active)
                if cell.state == BSState.SLEEP:
                    original_state = cell.state
                    cell.state = BSState.ACTIVE
                    baseline_power = cell.get_power_consumption()
                    cell.state = original_state
                    total_baseline_power += baseline_power
                else:
                    total_baseline_power += current_power
                    
        # Store for history
        self.energy_history.append(total_current_power)
        self.baseline_energy_history.append(total_baseline_power)
        
        if total_baseline_power > 0:
            savings = ((total_baseline_power - total_current_power) / total_baseline_power) * 100
            return max(0, savings)
        return 0
        
    def calculate_average_energy_savings(self) -> float:
        """Calculate average energy savings over history"""
        if len(self.energy_history) == len(self.baseline_energy_history) and len(self.energy_history) > 0:
            total_savings = 0
            valid_samples = 0
            
            for current, baseline in zip(self.energy_history, self.baseline_energy_history):
                if baseline > 0:
                    savings = ((baseline - current) / baseline) * 100
                    total_savings += savings
                    valid_samples += 1
                    
            return total_savings / valid_samples if valid_samples > 0 else 0
        return 0
        
    def get_energy_statistics(self) -> Dict:
        """Get comprehensive energy statistics"""
        if not self.energy_history:
            return {}
            
        current_powers = list(self.energy_history)
        baseline_powers = list(self.baseline_energy_history)
        
        savings_list = []
        for current, baseline in zip(current_powers, baseline_powers):
            if baseline > 0:
                savings_list.append(((baseline - current) / baseline) * 100)
                
        if not savings_list:
            return {}
            
        return {
            'average_savings': np.mean(savings_list),
            'max_savings': np.max(savings_list),
            'min_savings': np.min(savings_list),
            'std_savings': np.std(savings_list),
            'median_savings': np.median(savings_list),
            'total_energy_consumed': sum(current_powers),
            'total_energy_baseline': sum(baseline_powers),
            'samples': len(savings_list)
        }

class QoSMetrics:
    """Quality of Service metrics calculator"""
    
    def __init__(self, sinr_threshold=-6.0):
        self.sinr_threshold = sinr_threshold
        self.qos_history = deque(maxlen=1000)
        self.violation_count = 0
        self.total_measurements = 0
        
    def calculate_qos_metrics(self, cells, ues) -> Dict:
        """Calculate QoS metrics for all users"""
        sinr_values = []
        violations = 0
        satisfied_users = 0
        
        for ue in ues:
            if hasattr(ue, 'serving_cell') and ue.serving_cell:
                sinr = ue.serving_cell.calculate_sinr(ue.xyz, cells)
                sinr_values.append(sinr)
                
                if sinr >= self.sinr_threshold:
                    satisfied_users += 1
                else:
                    violations += 1
                    
        self.violation_count += violations
        self.total_measurements += len(ues)
        
        metrics = {
            'total_users': len(ues),
            'satisfied_users': satisfied_users,
            'violated_users': violations,
            'satisfaction_ratio': satisfied_users / len(ues) if len(ues) > 0 else 0,
            'violation_ratio': violations / len(ues) if len(ues) > 0 else 0,
            'average_sinr': np.mean(sinr_values) if sinr_values else 0,
            'min_sinr': np.min(sinr_values) if sinr_values else 0,
            'max_sinr': np.max(sinr_values) if sinr_values else 0,
            'sinr_std': np.std(sinr_values) if sinr_values else 0
        }
        
        self.qos_history.append(metrics)
        return metrics
        
    def get_qos_statistics(self) -> Dict:
        """Get cumulative QoS statistics"""
        if not self.qos_history:
            return {}
            
        satisfaction_ratios = [m['satisfaction_ratio'] for m in self.qos_history]
        violation_ratios = [m['violation_ratio'] for m in self.qos_history]
        avg_sinrs = [m['average_sinr'] for m in self.qos_history]
        
        return {
            'cumulative_violation_rate': self.violation_count / self.total_measurements if self.total_measurements > 0 else 0,
            'average_satisfaction_ratio': np.mean(satisfaction_ratios),
            'worst_satisfaction_ratio': np.min(satisfaction_ratios),
            'best_satisfaction_ratio': np.max(satisfaction_ratios),
            'average_network_sinr': np.mean(avg_sinrs),
            'sinr_stability': np.std(avg_sinrs),
            'total_violations': self.violation_count,
            'total_measurements': self.total_measurements
        }

class NetworkMetrics:
    """General network performance metrics"""
    
    def __init__(self):
        self.handover_count = 0
        self.sleep_events = 0
        self.wake_events = 0
        self.load_history = defaultdict(deque)
        
    def calculate_network_load_metrics(self, cells) -> Dict:
        """Calculate network load distribution metrics"""
        loads = []
        overloaded_cells = 0
        underloaded_cells = 0
        active_cells = 0
        sleeping_cells = 0
        
        for cell in cells:
            if hasattr(cell, 'get_load'):
                if cell.state == BSState.ACTIVE:
                    active_cells += 1
                    load = cell.get_load()
                    loads.append(load)
                    self.load_history[cell.i].append(load)
                    
                    if load > 0.8:
                        overloaded_cells += 1
                    elif load < 0.1:
                        underloaded_cells += 1
                else:
                    sleeping_cells += 1
                    
        return {
            'active_cells': active_cells,
            'sleeping_cells': sleeping_cells,
            'overloaded_cells': overloaded_cells,
            'underloaded_cells': underloaded_cells,
            'average_load': np.mean(loads) if loads else 0,
            'max_load': np.max(loads) if loads else 0,
            'min_load': np.min(loads) if loads else 0,
            'load_std': np.std(loads) if loads else 0,
            'load_balance_index': self._calculate_load_balance_index(loads)
        }
        
    def _calculate_load_balance_index(self, loads) -> float:
        """Calculate load balance index (0=perfectly balanced, 1=completely unbalanced)"""
        if not loads or len(loads) <= 1:
            return 0
            
        # Jain's fairness index adapted for load balancing
        sum_loads = sum(loads)
        sum_squares = sum(l**2 for l in loads)
        
        if sum_squares == 0:
            return 0
            
        fairness_index = (sum_loads**2) / (len(loads) * sum_squares)
        return 1 - fairness_index  # Convert to imbalance index
        
    def calculate_spectral_efficiency(self, cells, ues) -> Dict:
        """Calculate spectral efficiency metrics"""
        total_se = 0
        total_bandwidth = 0
        se_values = []
        
        for cell in cells:
            if cell.state == BSState.ACTIVE:
                cell_se = 0
                for ue_id in cell.attached:
                    if ue_id < len(ues):
                        ue = ues[ue_id]
                        sinr = cell.calculate_sinr(ue.xyz, cells)
                        if sinr > -20:  # Valid SINR range
                            se = log2(1 + 10**(sinr/10))
                            cell_se += se
                            se_values.append(se)
                            
                total_se += cell_se
                total_bandwidth += cell.bandwidth
                
        return {
            'total_spectral_efficiency': total_se,
            'average_spectral_efficiency': np.mean(se_values) if se_values else 0,
            'network_spectral_efficiency': total_se / (total_bandwidth / 1e6) if total_bandwidth > 0 else 0,
            'se_per_user': total_se / len(ues) if len(ues) > 0 else 0
        }
        
    def track_sleep_wake_events(self, cells):
        """Track sleep and wake events"""
        current_sleeping = set()
        
        for cell in cells:
            if hasattr(cell, 'state') and cell.state == BSState.SLEEP:
                current_sleeping.add(cell.i)
                
        # Compare with previous state (simplified tracking)
        if hasattr(self, 'previous_sleeping'):
            new_sleepers = current_sleeping - self.previous_sleeping
            new_wakers = self.previous_sleeping - current_sleeping
            
            self.sleep_events += len(new_sleepers)
            self.wake_events += len(new_wakers)
        else:
            self.sleep_events += len(current_sleeping)
            
        self.previous_sleeping = current_sleeping.copy()
        
        return {
            'current_sleeping_cells': len(current_sleeping),
            'total_sleep_events': self.sleep_events,
            'total_wake_events': self.wake_events,
            'sleeping_cell_ids': list(current_sleeping)
        }

class PerformanceAnalyzer:
    """Comprehensive performance analyzer combining all metrics"""
    
    def __init__(self, sinr_threshold=-6.0):
        self.energy_metrics = EnergyMetrics()
        self.qos_metrics = QoSMetrics(sinr_threshold)
        self.network_metrics = NetworkMetrics()
        self.analysis_history = []
        
    def analyze_timestep(self, cells, ues, timestamp=None) -> Dict:
        """Perform comprehensive analysis for current timestep"""
        analysis = {
            'timestamp': timestamp,
            'energy': self.energy_metrics.calculate_instantaneous_energy_savings(cells),
            'qos': self.qos_metrics.calculate_qos_metrics(cells, ues),
            'network_load': self.network_metrics.calculate_network_load_metrics(cells),
            'spectral_efficiency': self.network_metrics.calculate_spectral_efficiency(cells, ues),
            'sleep_wake': self.network_metrics.track_sleep_wake_events(cells)
        }
        
        self.analysis_history.append(analysis)
        return analysis
        
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.analysis_history:
            return {}
            
        report = {
            'simulation_summary': {
                'total_timesteps': len(self.analysis_history),
                'analysis_period': self.analysis_history[-1]['timestamp'] - self.analysis_history[0]['timestamp'] if len(self.analysis_history) > 1 else 0
            },
            'energy_performance': self.energy_metrics.get_energy_statistics(),
            'qos_performance': self.qos_metrics.get_qos_statistics(),
            'network_performance': self._analyze_network_performance(),
            'efficiency_metrics': self._calculate_efficiency_metrics()
        }
        
        return report
        
    def _analyze_network_performance(self) -> Dict:
        """Analyze network performance trends"""
        if not self.analysis_history:
            return {}
            
        sleep_counts = [a['sleep_wake']['current_sleeping_cells'] for a in self.analysis_history]
        avg_loads = [a['network_load']['average_load'] for a in self.analysis_history]
        se_values = [a['spectral_efficiency']['average_spectral_efficiency'] for a in self.analysis_history]
        
        return {
            'average_sleeping_cells': np.mean(sleep_counts),
            'max_sleeping_cells': np.max(sleep_counts),
            'sleep_utilization': np.mean(sleep_counts) / max(sleep_counts) if max(sleep_counts) > 0 else 0,
            'load_trend': np.polyfit(range(len(avg_loads)), avg_loads, 1)[0] if len(avg_loads) > 1 else 0,
            'spectral_efficiency_trend': np.polyfit(range(len(se_values)), se_values, 1)[0] if len(se_values) > 1 else 0,
            'network_stability': {
                'load_variance': np.var(avg_loads),
                'se_variance': np.var(se_values),
                'sleep_variance': np.var(sleep_counts)
            }
        }
        
    def _calculate_efficiency_metrics(self) -> Dict:
        """Calculate overall efficiency metrics"""
        if not self.analysis_history:
            return {}
            
        # Energy-QoS trade-off
        energy_savings = [a['energy'] for a in self.analysis_history]
        qos_satisfaction = [a['qos']['satisfaction_ratio'] for a in self.analysis_history]
        
        # Calculate correlation between energy savings and QoS
        if len(energy_savings) > 1 and len(qos_satisfaction) > 1:
            energy_qos_correlation = np.corrcoef(energy_savings, qos_satisfaction)[0, 1]
        else:
            energy_qos_correlation = 0
            
        return {
            'energy_qos_tradeoff': energy_qos_correlation,
            'efficiency_score': self._calculate_efficiency_score(),
            'performance_consistency': self._calculate_performance_consistency()
        }
        
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0-100)"""
        if not self.analysis_history:
            return 0
            
        # Weighted combination of metrics
        weights = {
            'energy_savings': 0.4,
            'qos_satisfaction': 0.4,
            'load_balance': 0.2
        }
        
        avg_energy = np.mean([a['energy'] for a in self.analysis_history])
        avg_qos = np.mean([a['qos']['satisfaction_ratio'] for a in self.analysis_history]) * 100
        avg_balance = (1 - np.mean([a['network_load']['load_balance_index'] for a in self.analysis_history])) * 100
        
        score = (weights['energy_savings'] * avg_energy + 
                weights['qos_satisfaction'] * avg_qos +
                weights['load_balance'] * avg_balance)
                
        return min(100, max(0, score))
        
    def _calculate_performance_consistency(self) -> float:
        """Calculate performance consistency score"""
        if not self.analysis_history:
            return 0
            
        # Lower variance = higher consistency
        energy_var = np.var([a['energy'] for a in self.analysis_history])
        qos_var = np.var([a['qos']['satisfaction_ratio'] for a in self.analysis_history])
        
        # Normalize and invert (high variance = low consistency)
        consistency = 1 / (1 + energy_var + qos_var * 100)
        return consistency * 100

# Utility functions for metric calculation
def calculate_jains_fairness_index(values):
    """Calculate Jain's fairness index"""
    if not values or len(values) <= 1:
        return 1.0
        
    sum_values = sum(values)
    sum_squares = sum(v**2 for v in values)
    
    if sum_squares == 0:
        return 1.0
        
    return (sum_values**2) / (len(values) * sum_squares)

def calculate_percentile_metrics(values, percentiles=[25, 50, 75, 90, 95]):
    """Calculate percentile metrics for a list of values"""
    if not values:
        return {}
        
    return {f'p{p}': np.percentile(values, p) for p in percentiles}

if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics calculation...")
    
    # Mock data for testing
    class MockCell:
        def __init__(self, cell_id, bs_type, state, power, load):
            self.i = cell_id
            self.bs_type = bs_type
            self.state = state
            self._power = power
            self._load = load
            self.bandwidth = 20e6
            self.attached = []
            
        def get_power_consumption(self):
            return self._power
            
        def get_load(self):
            return self._load
            
        def calculate_sinr(self, ue_pos, cells):
            return -5.0  # Mock SINR
    
    class MockUE:
        def __init__(self, ue_id):
            self.i = ue_id
            self.xyz = [1000, 1000, 2]
            self.serving_cell = None
    
    # Create test scenario
    cells = [
        MockCell(0, BSType.MACRO, BSState.ACTIVE, 200, 0.5),
        MockCell(1, BSType.MICRO, BSState.ACTIVE, 25, 0.3),
        MockCell(2, BSType.MICRO, BSState.SLEEP, 2, 0.0)
    ]
    
    ues = [MockUE(0), MockUE(1)]
    ues[0].serving_cell = cells[0]
    ues[1].serving_cell = cells[1]
    cells[0].attached = [0]
    cells[1].attached = [1]
    
    # Test analyzer
    analyzer = PerformanceAnalyzer()
    result = analyzer.analyze_timestep(cells, ues, timestamp=100.0)
    
    print("Analysis result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
        
    print("Metrics testing completed!")