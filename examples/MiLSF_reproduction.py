# examples/paper_reproduction.py
"""
Paper Reproduction Experiments
Reproduces the experimental scenarios from the MiLSF paper
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.simulation_config import get_config, PAPER_SCENARIOS
from scenarios.hetnet_scenarios import create_paper_scenario, run_comparative_study
from algorithms.milsf_ric import MiLSF_RIC
from utils.logger import MiLSF_Logger
from utils.metrics import PerformanceAnalyzer

class PaperReproduction:
    """Class to reproduce paper experiments"""
    
    def __init__(self, output_dir="data/results/paper_reproduction"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def run_scenario_1(self):
        """
        Scenario I: PPP vs MHCPP deployment comparison
        Figure 9 from the paper
        """
        print("\n=== Running Scenario I: PPP vs MHCPP Comparison ===")
        
        base_config = get_config('paper_reproduction')
        results = {'PPP': {}, 'MHCPP': {}}
        
        # Test both deployment types
        for deployment_type in ['PPP', 'MHCPP']:
            print(f"\nTesting {deployment_type} deployment...")
            
            config = base_config.copy()
            config['deployment_type'] = deployment_type
            
            try:
                sim = create_paper_scenario('scenario_1', config)
                sim.run(until=86400)  # 24 hours
                
                # Collect results
                if hasattr(sim, 'ric') and sim.ric:
                    ric = sim.ric
                    if ric.total_energy_without_sleeping > 0:
                        energy_savings = ((ric.total_energy_without_sleeping - ric.total_energy_with_sleeping) 
                                        / ric.total_energy_without_sleeping) * 100
                        
                        results[deployment_type] = {
                            'energy_savings': energy_savings,
                            'sleep_decisions': ric.sleep_decisions,
                            'wake_decisions': ric.wake_decisions,
                            'failed_reallocations': ric.failed_reallocations,
                            'avg_energy_savings': ric.calculate_energy_savings()
                        }
                        
                        print(f"  {deployment_type} Energy Savings: {energy_savings:.2f}%")
                        print(f"  {deployment_type} Sleep Decisions: {ric.sleep_decisions}")
                        
            except Exception as e:
                print(f"Error in {deployment_type}: {e}")
                results[deployment_type] = {'error': str(e)}
        
        self.results['scenario_1'] = results
        self._save_scenario_1_plot(results)
        
        return results
        
    def run_scenario_2(self):
        """
        Scenario II: User density impact
        Figure 10 from the paper
        """
        print("\n=== Running Scenario II: User Density Impact ===")
        
        base_config = get_config('paper_reproduction')
        user_counts = [10, 15, 25, 35, 50, 75, 100]
        results = {}
        
        for user_count in user_counts:
            print(f"\nTesting with {user_count} users...")
            
            config = base_config.copy()
            config['n_users'] = user_count
            
            try:
                sim = create_paper_scenario('scenario_2', config)
                sim.run(until=86400)
                
                # Collect results
                if hasattr(sim, 'ric') and sim.ric:
                    ric = sim.ric
                    if ric.total_energy_without_sleeping > 0:
                        energy_savings = ((ric.total_energy_without_sleeping - ric.total_energy_with_sleeping) 
                                        / ric.total_energy_without_sleeping) * 100
                        
                        results[user_count] = {
                            'energy_savings': energy_savings,
                            'sleep_decisions': ric.sleep_decisions,
                            'sleeping_cells': len(ric.sleeping_cells),
                            'failed_reallocations': ric.failed_reallocations
                        }
                        
                        print(f"  {user_count} users: {energy_savings:.2f}% energy savings")
                        
            except Exception as e:
                print(f"Error with {user_count} users: {e}")
                results[user_count] = {'error': str(e)}
        
        self.results['scenario_2'] = results
        self._save_scenario_2_plot(results)
        
        return results
        
    def run_scenario_3(self):
        """
        Scenario III: SINR threshold impact
        Figure 11 from the paper
        """
        print("\n=== Running Scenario III: SINR Threshold Impact ===")
        
        base_config = get_config('paper_reproduction')
        sinr_thresholds = [-12, -9, -6, -3, 0]
        results = {}
        
        for sinr_threshold in sinr_thresholds:
            print(f"\nTesting SINR threshold: {sinr_threshold} dB...")
            
            config = base_config.copy()
            config['sinr_threshold'] = sinr_threshold
            
            try:
                sim = create_paper_scenario('scenario_3', config)
                
                # Update RIC SINR threshold
                if hasattr(sim, 'ric') and sim.ric:
                    sim.ric.sinr_threshold = sinr_threshold
                
                sim.run(until=86400)
                
                # Collect results
                if hasattr(sim, 'ric') and sim.ric:
                    ric = sim.ric
                    if ric.total_energy_without_sleeping > 0:
                        energy_savings = ((ric.total_energy_without_sleeping - ric.total_energy_with_sleeping) 
                                        / ric.total_energy_without_sleeping) * 100
                        
                        results[sinr_threshold] = {
                            'energy_savings': energy_savings,
                            'sleep_decisions': ric.sleep_decisions,
                            'failed_reallocations': ric.failed_reallocations
                        }
                        
                        print(f"  SINR {sinr_threshold} dB: {energy_savings:.2f}% energy savings")
                        
            except Exception as e:
                print(f"Error with SINR {sinr_threshold}: {e}")
                results[sinr_threshold] = {'error': str(e)}
        
        self.results['scenario_3'] = results
        self._save_scenario_3_plot(results)
        
        return results
        
    def run_scenario_4(self):
        """
        Scenario IV: Sleeping cells vs energy savings relationship
        Figure 12 from the paper
        """
        print("\n=== Running Scenario IV: Sleeping Cells vs Energy Savings ===")
        
        base_config = get_config('paper_reproduction')
        results = {}
        
        # Run simulation and track sleeping cells over time
        sim = create_paper_scenario('scenario_4', base_config)
        
        # Track metrics during simulation
        time_points = []
        energy_savings_points = []
        sleeping_cells_points = []
        
        # Run simulation with periodic sampling
        total_time = 86400  # 24 hours
        sample_interval = 3600  # Every hour
        
        for t in range(0, total_time, sample_interval):
            sim.run(until=t + sample_interval)
            
            if hasattr(sim, 'ric') and sim.ric:
                ric = sim.ric
                current_energy_savings = ric.calculate_energy_savings()
                current_sleeping_cells = len(ric.sleeping_cells)
                
                time_points.append(t / 3600)  # Convert to hours
                energy_savings_points.append(current_energy_savings)
                sleeping_cells_points.append(current_sleeping_cells)
                
        results = {
            'time_hours': time_points,
            'energy_savings': energy_savings_points,
            'sleeping_cells': sleeping_cells_points
        }
        
        self.results['scenario_4'] = results
        self._save_scenario_4_plot(results)
        
        return results
        
    def _save_scenario_1_plot(self, results):
        """Save Scenario 1 plot (Figure 9 reproduction)"""
        try:
            import matplotlib.pyplot as plt
            
            deployments = ['PPP', 'MHCPP']
            energy_savings = []
            
            for dep in deployments:
                if 'error' not in results[dep]:
                    energy_savings.append(results[dep]['energy_savings'])
                else:
                    energy_savings.append(0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(deployments, energy_savings, color=['lightcoral', 'skyblue'])
            
            ax.set_ylabel('Energy Saving Percentage (%)')
            ax.set_title('MiLSF Energy Savings: PPP vs MHCPP Deployment')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, energy_savings):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.2f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'scenario_1_ppp_vs_mhcpp.png', dpi=300)
            plt.savefig(self.output_dir / 'scenario_1_ppp_vs_mhcpp.pdf')
            plt.close()
            
            print(f"Scenario 1 plot saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error saving Scenario 1 plot: {e}")
            
    def _save_scenario_2_plot(self, results):
        """Save Scenario 2 plot (Figure 10 reproduction)"""
        try:
            import matplotlib.pyplot as plt
            
            user_counts = []
            energy_savings = []
            
            for user_count, result in results.items():
                if 'error' not in result:
                    user_counts.append(user_count)
                    energy_savings.append(result['energy_savings'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(user_counts, energy_savings, 'bo-', linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Users')
            ax.set_ylabel('Energy Saving Percentage (%)')
            ax.set_title('MiLSF Energy Savings vs User Density')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(user_counts) > 1:
                z = np.polyfit(user_counts, energy_savings, 1)
                p = np.poly1d(z)
                ax.plot(user_counts, p(user_counts), "r--", alpha=0.8, label='Trend')
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'scenario_2_user_density.png', dpi=300)
            plt.savefig(self.output_dir / 'scenario_2_user_density.pdf')
            plt.close()
            
            print(f"Scenario 2 plot saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error saving Scenario 2 plot: {e}")
            
    def _save_scenario_3_plot(self, results):
        """Save Scenario 3 plot (Figure 11 reproduction)"""
        try:
            import matplotlib.pyplot as plt
            
            sinr_thresholds = []
            energy_savings = []
            
            for sinr_threshold, result in results.items():
                if 'error' not in result:
                    sinr_thresholds.append(sinr_threshold)
                    energy_savings.append(result['energy_savings'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sinr_thresholds, energy_savings, 'go-', linewidth=2, markersize=8)
            
            ax.set_xlabel('SINR Threshold (dB)')
            ax.set_ylabel('Energy Saving Percentage (%)')
            ax.set_title('MiLSF Energy Savings vs SINR Threshold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'scenario_3_sinr_threshold.png', dpi=300)
            plt.savefig(self.output_dir / 'scenario_3_sinr_threshold.pdf')
            plt.close()
            
            print(f"Scenario 3 plot saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error saving Scenario 3 plot: {e}")
            
    def _save_scenario_4_plot(self, results):
        """Save Scenario 4 plot (Figure 12 reproduction)"""
        try:
            import matplotlib.pyplot as plt
            
            time_hours = results['time_hours']
            energy_savings = results['energy_savings']
            sleeping_cells = results['sleeping_cells']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Energy savings over time
            ax1.plot(time_hours, energy_savings, 'b-', linewidth=2, label='Energy Savings')
            ax1.set_ylabel('Energy Savings (%)')
            ax1.set_title('MiLSF Performance Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Sleeping cells over time
            ax2.plot(time_hours, sleeping_cells, 'r-', linewidth=2, label='Sleeping Cells')
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Number of Sleeping Cells')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Mark low-load period
            for ax in [ax1, ax2]:
                ax.axvspan(22, 24, alpha=0.2, color='gray', label='Low-load period')
                ax.axvspan(0, 6, alpha=0.2, color='gray')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'scenario_4_time_analysis.png', dpi=300)
            plt.savefig(self.output_dir / 'scenario_4_time_analysis.pdf')
            plt.close()
            
            print(f"Scenario 4 plot saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error saving Scenario 4 plot: {e}")
            
    def run_all_scenarios(self):
        """Run all paper reproduction scenarios"""
        print("="*60)
        print("STARTING PAPER REPRODUCTION EXPERIMENTS")
        print("="*60)
        
        start_time = time.time()
        
        # Run all scenarios
        self.run_scenario_1()
        self.run_scenario_2() 
        self.run_scenario_3()
        self.run_scenario_4()
        
        end_time = time.time()
        
        # Generate summary report
        self._generate_summary_report(end_time - start_time)
        
        return self.results
        
    def _generate_summary_report(self, total_time):
        """Generate comprehensive summary report"""
        print(f"\n{'='*60}")
        print("PAPER REPRODUCTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total execution time: {total_time:.1f} seconds")
        
        # Summary for each scenario
        for scenario_name, results in self.results.items():
            print(f"\n{scenario_name.upper()}:")
            
            if scenario_name == 'scenario_1':
                for deployment, result in results.items():
                    if 'error' not in result:
                        print(f"  {deployment}: {result['energy_savings']:.2f}% energy savings")
                    else:
                        print(f"  {deployment}: ERROR")
                        
            elif scenario_name == 'scenario_2':
                successful_runs = [r for r in results.values() if 'error' not in r]
                if successful_runs:
                    avg_savings = np.mean([r['energy_savings'] for r in successful_runs])
                    max_savings = max([r['energy_savings'] for r in successful_runs])
                    print(f"  Average energy savings: {avg_savings:.2f}%")
                    print(f"  Maximum energy savings: {max_savings:.2f}%")
                    print(f"  Successful runs: {len(successful_runs)}/{len(results)}")
                    
            elif scenario_name == 'scenario_3':
                successful_runs = [r for r in results.values() if 'error' not in r]
                if successful_runs:
                    avg_savings = np.mean([r['energy_savings'] for r in successful_runs])
                    print(f"  Average energy savings: {avg_savings:.2f}%")
                    print(f"  Successful runs: {len(successful_runs)}/{len(results)}")
                    
            elif scenario_name == 'scenario_4':
                if 'energy_savings' in results:
                    max_savings = max(results['energy_savings'])
                    max_sleeping = max(results['sleeping_cells'])
                    print(f"  Peak energy savings: {max_savings:.2f}%")
                    print(f"  Peak sleeping cells: {max_sleeping}")
        
        # Save summary to file
        summary_file = self.output_dir / 'reproduction_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("MiLSF Paper Reproduction Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Execution time: {total_time:.1f} seconds\n\n")
            
            for scenario_name, results in self.results.items():
                f.write(f"{scenario_name}:\n")
                f.write(str(results) + "\n\n")
        
        print(f"\nDetailed results saved to: {self.output_dir}")
        print(f"Summary saved to: {summary_file}")

def run_paper_experiments(scenarios=['all'], output_dir=None):
    """
    Main function to run paper reproduction experiments
    
    Args:
        scenarios: List of scenarios to run or ['all']
        output_dir: Output directory for results
    """
    if output_dir is None:
        output_dir = "data/results/paper_reproduction"
    
    reproducer = PaperReproduction(output_dir)
    
    if 'all' in scenarios:
        return reproducer.run_all_scenarios()
    else:
        results = {}
        for scenario in scenarios:
            if scenario == '1':
                results['scenario_1'] = reproducer.run_scenario_1()
            elif scenario == '2':
                results['scenario_2'] = reproducer.run_scenario_2()
            elif scenario == '3':
                results['scenario_3'] = reproducer.run_scenario_3()
            elif scenario == '4':
                results['scenario_4'] = reproducer.run_scenario_4()
            else:
                print(f"Unknown scenario: {scenario}")
        
        return results

def compare_with_baselines(config=None):
    """Compare MiLSF with baseline strategies"""
    print("\n=== Baseline Comparison ===")
    
    if config is None:
        from config.simulation_config import get_config
        config = get_config('paper_reproduction')
    
    # Baseline strategies simulation would go here
    # For now, we'll use the theoretical comparison from the paper
    
    baselines = {
        'No Sleeping': 0.0,
        'Random Sleep (RS)': 4.63,
        'Random Reallocation (RRU)': 3.44,
        'Closest User Reallocation (CUR)': 3.67,
        'Closest BS Sleep First (CBSSF)': 3.27,
        'MiLSF (Proposed)': 11.26  # Expected from paper
    }
    
    print("Baseline comparison (Energy Savings %):")
    for strategy, savings in baselines.items():
        print(f"  {strategy}: {savings:.2f}%")
    
    return baselines

def validate_reproduction_accuracy(results, paper_results=None):
    """Validate reproduction accuracy against paper results"""
    if paper_results is None:
        # Expected results from paper (approximate)
        paper_results = {
            'scenario_1': {'MHCPP': 11.26, 'PPP': 5.82},
            'scenario_2': {'25_users': 11.26},  # Reference point
            'scenario_3': {'-6_dB': 11.26},     # Reference point
        }
    
    print("\n=== Reproduction Validation ===")
    
    validation_report = {}
    
    for scenario, paper_result in paper_results.items():
        if scenario in results:
            sim_result = results[scenario]
            
            if scenario == 'scenario_1':
                for deployment in ['PPP', 'MHCPP']:
                    if deployment in sim_result and 'error' not in sim_result[deployment]:
                        paper_val = paper_result[deployment]
                        sim_val = sim_result[deployment]['energy_savings']
                        error_pct = abs(sim_val - paper_val) / paper_val * 100
                        
                        validation_report[f"{scenario}_{deployment}"] = {
                            'paper': paper_val,
                            'simulation': sim_val,
                            'error_percent': error_pct
                        }
                        
                        print(f"{scenario} {deployment}:")
                        print(f"  Paper: {paper_val:.2f}%")
                        print(f"  Simulation: {sim_val:.2f}%")
                        print(f"  Error: {error_pct:.1f}%")
    
    return validation_report

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Reproduce MiLSF paper experiments')
    parser.add_argument('--scenario', '-s', choices=['1', '2', '3', '4', 'all'], 
                       default='all', help='Scenario to run')
    parser.add_argument('--output', '-o', type=str, 
                       default='data/results/paper_reproduction',
                       help='Output directory')
    parser.add_argument('--validate', action='store_true',
                       help='Validate results against paper')
    parser.add_argument('--baselines', action='store_true',
                       help='Compare with baseline strategies')
    
    args = parser.parse_args()
    
    # Run experiments
    if args.scenario == 'all':
        scenarios = ['all']
    else:
        scenarios = [args.scenario]
    
    print("Starting MiLSF paper reproduction...")
    print(f"Scenarios: {scenarios}")
    print(f"Output directory: {args.output}")
    
    # Run main experiments
    results = run_paper_experiments(scenarios, args.output)
    
    # Optional validation
    if args.validate:
        validate_reproduction_accuracy(results)
    
    # Optional baseline comparison
    if args.baselines:
        compare_with_baselines()
    
    print("\nPaper reproduction completed!")
    
    return results

if __name__ == "__main__":
    main()