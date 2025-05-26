# src/utils/logger.py
"""
Logging utilities for MiLSF simulation
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from AIMM_simulator import Logger
from ..core.enhanced_cell import BSState, BSType

class MiLSF_Logger(Logger):
    """Enhanced logger for MiLSF strategy performance"""
    
    def __init__(self, sim, logging_interval=60.0, output_dir="data/results", 
                 filename_prefix="milsf_simulation"):
        super().__init__(sim, logging_interval)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_filename = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
        self.json_filename = self.output_dir / f"{filename_prefix}_{timestamp}_summary.json"
        
        # Summary statistics
        self.summary_stats = {
            'simulation_start': time.time(),
            'total_energy_savings': 0.0,
            'max_energy_savings': 0.0,
            'avg_energy_savings': 0.0,
            'sleep_events': 0,
            'wake_events': 0,
            'peak_sleeping_cells': 0,
            'qos_violations': 0,
            'energy_history': [],
            'sleep_history': []
        }
        
    def loop(self):
        """Main logging loop"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            self.csv_writer = csv.writer(csvfile, delimiter='\t')
            
            # Write header
            self.csv_writer.writerow([
                'time', 'cell_id', 'bs_type', 'state', 'load', 'power_W', 
                'users', 'sinr_avg', 'throughput', 'energy_savings_%'
            ])
            
            while True:
                self.log_timestep()
                yield self.sim.wait(self.logging_interval)
                
    def log_timestep(self):
        """Log data for current timestep"""
        current_time = self.sim.env.now
        
        # Calculate network-wide metrics
        total_active_power = 0
        total_current_power = 0
        sleeping_cells = 0
        qos_violations = 0
        
        for cell in self.sim.cells:
            if hasattr(cell, 'bs_type'):
                # Cell metrics
                load = cell.get_load()
                power = cell.get_power_consumption()
                n_users = len(cell.attached)
                
                # Calculate average SINR for attached users
                sinr_values = []
                throughput = 0
                
                for ue_id in cell.attached:
                    ue = self.sim.UEs[ue_id]
                    sinr = cell.calculate_sinr(ue.xyz, self.sim.cells)
                    sinr_values.append(sinr)
                    
                    # Check QoS violations
                    if sinr < -6:  # SINR threshold
                        qos_violations += 1
                        
                    # Estimate throughput (simplified)
                    if sinr > -6:
                        throughput += cell.bandwidth * np.log2(1 + 10**(sinr/10)) / 1e6  # Mbps
                
                avg_sinr = sum(sinr_values) / len(sinr_values) if sinr_values else 0
                
                # Track power for energy savings calculation
                total_current_power += power
                
                if cell.state == BSState.SLEEP:
                    sleeping_cells += 1
                    # Calculate what power would be if active
                    original_state = cell.state
                    cell.state = BSState.ACTIVE
                    active_power = cell.get_power_consumption()
                    cell.state = original_state
                    total_active_power += active_power
                else:
                    total_active_power += power
                    
                # Calculate instantaneous energy savings
                energy_savings = 0
                if total_active_power > 0:
                    energy_savings = ((total_active_power - total_current_power) / total_active_power) * 100
                
                # Write to CSV
                self.csv_writer.writerow([
                    f'{current_time:.1f}', cell.i, cell.bs_type.value, cell.state.value,
                    f'{load:.3f}', f'{power:.1f}', n_users, f'{avg_sinr:.1f}',
                    f'{throughput:.2f}', f'{energy_savings:.2f}'
                ])
                
        # Update summary statistics
        if total_active_power > 0:
            current_savings = ((total_active_power - total_current_power) / total_active_power) * 100
            self.summary_stats['energy_history'].append(current_savings)
            self.summary_stats['max_energy_savings'] = max(
                self.summary_stats['max_energy_savings'], current_savings
            )
            
        self.summary_stats['sleep_history'].append(sleeping_cells)
        self.summary_stats['peak_sleeping_cells'] = max(
            self.summary_stats['peak_sleeping_cells'], sleeping_cells
        )
        self.summary_stats['qos_violations'] += qos_violations
        
    def finalize(self):
        """Finalize logging and save summary"""
        # Calculate final statistics
        if self.summary_stats['energy_history']:
            self.summary_stats['avg_energy_savings'] = (
                sum(self.summary_stats['energy_history']) / len(self.summary_stats['energy_history'])
            )
            
        self.summary_stats['simulation_duration'] = time.time() - self.summary_stats['simulation_start']
        
        # Save summary to JSON
        with open(self.json_filename, 'w') as f:
            json.dump(self.summary_stats, f, indent=2)
            
        print(f"\nLogging completed:")
        print(f"  CSV data: {self.csv_filename}")
        print(f"  Summary: {self.json_filename}")
        print(f"  Average energy savings: {self.summary_stats['avg_energy_savings']:.2f}%")

class DetailedLogger(MiLSF_Logger):
    """More detailed logger with additional metrics"""
    
    def __init__(self, sim, logging_interval=60.0, output_dir="data/results", 
                 filename_prefix="detailed_milsf"):
        super().__init__(sim, logging_interval, output_dir, filename_prefix)
        
        # Additional detailed logs
        self.ue_log_filename = self.output_dir / f"{filename_prefix}_ue_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.handover_log_filename = self.output_dir / f"{filename_prefix}_handovers_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Track handovers
        self.previous_attachments = {}
        
    def loop(self):
        """Enhanced logging loop with UE details"""
        with open(self.csv_filename, 'w', newline='') as csvfile, \
             open(self.ue_log_filename, 'w', newline='') as ue_file, \
             open(self.handover_log_filename, 'w', newline='') as ho_file:
            
            self.csv_writer = csv.writer(csvfile, delimiter='\t')
            self.ue_writer = csv.writer(ue_file, delimiter='\t')
            self.ho_writer = csv.writer(ho_file, delimiter='\t')
            
            # Write headers
            self.csv_writer.writerow([
                'time', 'cell_id', 'bs_type', 'state', 'load', 'power_W', 
                'users', 'sinr_avg', 'throughput', 'energy_savings_%',
                'interference', 'spectral_efficiency'
            ])
            
            self.ue_writer.writerow([
                'time', 'ue_id', 'serving_cell', 'cell_type', 'sinr_dB',
                'traffic_rate', 'position_x', 'position_y', 'qos_satisfied'
            ])
            
            self.ho_writer.writerow([
                'time', 'ue_id', 'source_cell', 'target_cell', 'reason'
            ])
            
            while True:
                self.log_detailed_timestep()
                yield self.sim.wait(self.logging_interval)
                
    def log_detailed_timestep(self):
        """Log detailed timestep data"""
        current_time = self.sim.env.now
        
        # Log cell data (same as parent)
        self.log_timestep()
        
        # Log UE data
        for ue in self.sim.UEs:
            if hasattr(ue, 'serving_cell') and ue.serving_cell:
                serving_cell = ue.serving_cell
                sinr = serving_cell.calculate_sinr(ue.xyz, self.sim.cells)
                traffic_rate = getattr(ue, 'current_traffic_rate', 1.0)
                qos_satisfied = sinr >= -6  # SINR threshold
                
                self.ue_writer.writerow([
                    f'{current_time:.1f}', ue.i, serving_cell.i, serving_cell.bs_type.value,
                    f'{sinr:.1f}', f'{traffic_rate:.3f}', 
                    f'{ue.xyz[0]:.1f}', f'{ue.xyz[1]:.1f}', qos_satisfied
                ])
                
                # Detect handovers
                if ue.i in self.previous_attachments:
                    if self.previous_attachments[ue.i] != serving_cell.i:
                        self.ho_writer.writerow([
                            f'{current_time:.1f}', ue.i, 
                            self.previous_attachments[ue.i], serving_cell.i,
                            'signal_quality'
                        ])
                        
                self.previous_attachments[ue.i] = serving_cell.i

class RealTimeLogger:
    """Real-time logger for live monitoring"""
    
    def __init__(self, sim, update_interval=10.0):
        self.sim = sim
        self.update_interval = update_interval
        self.metrics_history = []
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        import threading
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def _monitor_loop(self):
        """Real-time monitoring loop"""
        while True:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only last 100 measurements
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
                
            time.sleep(self.update_interval)
            
    def _collect_metrics(self):
        """Collect current network metrics"""
        metrics = {
            'timestamp': time.time(),
            'simulation_time': self.sim.env.now,
            'active_cells': 0,
            'sleeping_cells': 0,
            'total_power': 0,
            'total_users': len(self.sim.UEs),
            'attached_users': 0
        }
        
        for cell in self.sim.cells:
            if hasattr(cell, 'state'):
                if cell.state == BSState.ACTIVE:
                    metrics['active_cells'] += 1
                else:
                    metrics['sleeping_cells'] += 1
                    
                metrics['total_power'] += cell.get_power_consumption()
                metrics['attached_users'] += len(cell.attached)
                
        return metrics
        
    def get_current_metrics(self):
        """Get current metrics for display"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}
        
    def get_metrics_history(self, last_n=10):
        """Get recent metrics history"""
        return self.metrics_history[-last_n:]

class CSVAnalyzer:
    """Utility for analyzing CSV log files"""
    
    def __init__(self, csv_filename):
        self.csv_filename = Path(csv_filename)
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load CSV data"""
        try:
            import pandas as pd
            self.data = pd.read_csv(self.csv_filename, sep='\t')
            print(f"Loaded {len(self.data)} records from {self.csv_filename}")
        except ImportError:
            print("pandas not available, using basic CSV reader")
            self._load_data_basic()
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def _load_data_basic(self):
        """Basic CSV loading without pandas"""
        self.data = []
        with open(self.csv_filename, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                self.data.append(row)
                
    def calculate_energy_savings_stats(self):
        """Calculate energy savings statistics"""
        if self.data is None:
            return {}
            
        try:
            import pandas as pd
            
            # Convert to numeric
            energy_savings = pd.to_numeric(self.data['energy_savings_%'], errors='coerce')
            energy_savings = energy_savings.dropna()
            
            stats = {
                'mean': energy_savings.mean(),
                'max': energy_savings.max(),
                'min': energy_savings.min(),
                'std': energy_savings.std(),
                'median': energy_savings.median()
            }
            
            return stats
            
        except:
            # Fallback to basic calculation
            energy_values = []
            for row in self.data:
                try:
                    value = float(row['energy_savings_%'])
                    energy_values.append(value)
                except:
                    continue
                    
            if energy_values:
                return {
                    'mean': sum(energy_values) / len(energy_values),
                    'max': max(energy_values),
                    'min': min(energy_values),
                    'count': len(energy_values)
                }
            return {}
            
    def get_sleep_events(self):
        """Extract sleep/wake events"""
        events = []
        
        for row in self.data:
            if row.get('state') == 'sleep' and row.get('bs_type') == 'micro':
                events.append({
                    'time': float(row['time']),
                    'cell_id': int(row['cell_id']),
                    'event': 'sleep'
                })
            elif row.get('state') == 'active' and row.get('bs_type') == 'micro':
                # Check if this was previously sleeping (simplified)
                events.append({
                    'time': float(row['time']),
                    'cell_id': int(row['cell_id']), 
                    'event': 'wake'
                })
                
        return events
        
    def plot_energy_savings(self, output_file=None):
        """Plot energy savings over time"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Convert data
            df = pd.DataFrame(self.data)
            df['time'] = pd.to_numeric(df['time'])
            df['energy_savings_%'] = pd.to_numeric(df['energy_savings_%'])
            
            # Group by time and take mean
            time_grouped = df.groupby('time')['energy_savings_%'].mean()
            
            plt.figure(figsize=(12, 6))
            plt.plot(time_grouped.index / 3600, time_grouped.values, linewidth=2)
            plt.xlabel('Time (hours)')
            plt.ylabel('Energy Savings (%)')
            plt.title('MiLSF Energy Savings Over Time')
            plt.grid(True, alpha=0.3)
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {output_file}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Plotting failed: {e}")

class LogFormatter:
    """Utility for formatting log outputs"""
    
    @staticmethod
    def format_simulation_summary(summary_file):
        """Format simulation summary for display"""
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
            print("=" * 50)
            print("SIMULATION SUMMARY")
            print("=" * 50)
            print(f"Duration: {data.get('simulation_duration', 0):.1f} seconds")
            print(f"Average Energy Savings: {data.get('avg_energy_savings', 0):.2f}%")
            print(f"Maximum Energy Savings: {data.get('max_energy_savings', 0):.2f}%")
            print(f"Peak Sleeping Cells: {data.get('peak_sleeping_cells', 0)}")
            print(f"QoS Violations: {data.get('qos_violations', 0)}")
            print(f"Sleep Events: {data.get('sleep_events', 0)}")
            print(f"Wake Events: {data.get('wake_events', 0)}")
            
            if 'energy_history' in data and len(data['energy_history']) > 0:
                energy_hist = data['energy_history']
                print(f"Energy Savings Std Dev: {np.std(energy_hist):.2f}%")
                
            print("=" * 50)
            
        except Exception as e:
            print(f"Error formatting summary: {e}")
            
    @staticmethod
    def export_to_excel(csv_files, output_file):
        """Export multiple CSV files to Excel sheets"""
        try:
            import pandas as pd
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file, sep='\t')
                    sheet_name = Path(csv_file).stem[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
            print(f"Excel file created: {output_file}")
            
        except ImportError:
            print("pandas and openpyxl required for Excel export")
        except Exception as e:
            print(f"Excel export failed: {e}")

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

# Example usage and testing
if __name__ == "__main__":
    # Test CSVAnalyzer with sample data
    sample_data = [
        ['time', 'cell_id', 'bs_type', 'state', 'energy_savings_%'],
        ['0.0', '0', 'macro', 'active', '0.0'],
        ['60.0', '1', 'micro', 'sleep', '8.5'],
        ['120.0', '1', 'micro', 'sleep', '8.2'],
        ['180.0', '1', 'micro', 'active', '0.0']
    ]
    
    # Create sample CSV for testing
    test_file = Path("test_sample.csv")
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sample_data)
        
    # Test analyzer
    analyzer = CSVAnalyzer(test_file)
    stats = analyzer.calculate_energy_savings_stats()
    print("Energy savings stats:", stats)
    
    events = analyzer.get_sleep_events()
    print("Sleep events:", events)
    
    # Clean up
    test_file.unlink()
    print("Logger utilities test completed!")