# examples/basic_milsf_demo.py
"""
Basic MiLSF Demonstration
Simple example showing how to run MiLSF simulation
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from AIMM_simulator import Sim, Logger
from core.enhanced_cell import EnhancedCell, BSType
from core.traffic_aware_ue import TrafficAwareUE  
from algorithms.milsf_ric import MiLSF_RIC

class BasicLogger(Logger):
    """Simple logger for demonstration"""
    
    def loop(self):
        self.f.write('#time\tcell_id\tbs_type\tstate\tload\tpower_W\tusers\n')
        
        while True:
            for cell in self.sim.cells:
                if hasattr(cell, 'bs_type'):
                    load = cell.get_load()
                    power = cell.get_power_consumption()
                    n_users = len(cell.attached)
                    
                    self.f.write(f'{self.sim.env.now:.1f}\t{cell.i}\t{cell.bs_type.value}\t'
                               f'{cell.state.value}\t{load:.3f}\t{power:.1f}\t{n_users}\n')
                               
            yield self.sim.wait(self.logging_interval)

def create_simple_hetnet():
    """Create a simple heterogeneous network for demonstration"""
    
    # Create simulation
    sim = Sim(params={'fc_GHz': 2.4, 'h_UT': 2.0, 'h_BS': 25.0})
    
    print("Creating simple heterogeneous network...")
    
    # Deploy 3 Macro cells in triangular formation
    macro_positions = [
        (2000, 2000),  # Center-left
        (6000, 2000),  # Center-right  
        (4000, 5000)   # Top-center
    ]
    
    for i, (x, y) in enumerate(macro_positions):
        cell = EnhancedCell(
            sim,
            bs_type=BSType.MACRO,
            xyz=(x, y, 25.0),
            n_subbands=1,
            verbosity=0
        )
        print(f"  Deployed Macro BS {cell.i} at ({x}, {y})")
        
    # Deploy 5 Micro cells randomly
    micro_positions = [
        (3000, 3000),
        (5000, 3000), 
        (2500, 4000),
        (5500, 4000),
        (4000, 2500)
    ]
    
    for i, (x, y) in enumerate(micro_positions):
        cell = EnhancedCell(
            sim,
            bs_type=BSType.MICRO,
            xyz=(x, y, 10.0),
            n_subbands=1,
            verbosity=0
        )
        print(f"  Deployed Micro BS {cell.i} at ({x}, {y})")
        
    # Deploy 15 UEs randomly
    print("Deploying UEs...")
    np.random.seed(42)  # For reproducible results
    
    for i in range(15):
        x = np.random.uniform(1500, 6500)
        y = np.random.uniform(1500, 5500)
        
        ue = TrafficAwareUE(sim, xyz=(x, y, 2.0), verbosity=0)
        
        # Attach to best cell
        best_cell = None
        best_sinr = -float('inf')
        
        for cell in sim.cells:
            sinr = cell.calculate_sinr(ue.xyz, sim.cells)
            if sinr > best_sinr and sinr > -6:  # SINR threshold
                best_sinr = sinr
                best_cell = cell
                
        if best_cell:
            ue.attach(best_cell)
            print(f"  UE {i} attached to {best_cell.bs_type.value} BS {best_cell.i} (SINR: {best_sinr:.1f} dB)")
        else:
            print(f"  UE {i} could not attach to any cell")
            
    return sim

def run_basic_demo():
    """Run basic MiLSF demonstration"""
    
    print("=== MiLSF Basic Demonstration ===\n")
    
    # Create network
    sim = create_simple_hetnet()
    
    # Print initial network status
    print(f"\nNetwork created with:")
    macro_cells = [c for c in sim.cells if c.bs_type == BSType.MACRO]
    micro_cells = [c for c in sim.cells if c.bs_type == BSType.MICRO]
    print(f"  - {len(macro_cells)} Macro cells")
    print(f"  - {len(micro_cells)} Micro cells") 
    print(f"  - {len(sim.UEs)} UEs")
    
    # Show initial loads
    print(f"\nInitial cell loads:")
    for cell in sim.cells:
        load = cell.get_load()
        power = cell.get_power_consumption()
        users = len(cell.attached)
        print(f"  {cell.bs_type.value} BS {cell.i}: Load={load:.3f}, Power={power:.1f}W, Users={users}")
        
    # Add MiLSF RIC
    print(f"\nAdding MiLSF RIC...")
    ric = MiLSF_RIC(
        sim, 
        interval=60.0,  # Check every minute
        low_load_start=22,  # 10 PM
        low_load_end=6,     # 6 AM
        verbosity=1
    )
    sim.add_ric(ric)
    
    # Add logger
    logger = BasicLogger(sim, logging_interval=300.0)  # Log every 5 minutes
    sim.add_logger(logger)
    
    print(f"Starting simulation...")
    print(f"Low-load period: {ric.low_load_start}:00 - {ric.low_load_end}:00")
    print(f"Simulation will run for 24 hours (86400 seconds)")
    print(f"Watch for MiLSF decisions during low-load period!\n")
    
    # Run simulation for 24 hours
    sim.run(until=86400)
    
    print(f"\n=== Simulation Complete ===")
    
    # Print final statistics
    print_final_statistics(sim, ric)

def print_final_statistics(sim, ric):
    """Print final simulation statistics"""
    
    print(f"\nFinal Network Status:")
    
    total_power = 0
    active_cells = 0
    sleeping_cells = 0
    
    for cell in sim.cells:
        power = cell.get_power_consumption()
        total_power += power
        
        if hasattr(cell, 'state'):
            if cell.state.value == 'active':
                active_cells += 1
            else:
                sleeping_cells += 1
                
        load = cell.get_load()
        users = len(cell.attached)
        print(f"  {cell.bs_type.value} BS {cell.i}: {cell.state.value}, "
              f"Load={load:.3f}, Power={power:.1f}W, Users={users}")
              
    print(f"\nNetwork Summary:")
    print(f"  Total Power Consumption: {total_power:.1f} W")
    print(f"  Active Cells: {active_cells}")
    print(f"  Sleeping Cells: {sleeping_cells}")
    
    # Print RIC statistics
    if hasattr(ric, 'total_energy_without_sleeping') and ric.total_energy_without_sleeping > 0:
        total_savings = ((ric.total_energy_without_sleeping - ric.total_energy_with_sleeping) 
                        / ric.total_energy_without_sleeping) * 100
        print(f"\nMiLSF Performance:")
        print(f"  Total Energy Savings: {total_savings:.2f}%")
        print(f"  Sleep Decisions Made: {ric.sleep_decisions}")
        print(f"  Wake Decisions Made: {ric.wake_decisions}")
        print(f"  Failed Reallocations: {ric.failed_reallocations}")

def simulate_time_progression():
    """Demonstrate time progression and traffic changes"""
    
    print("\n=== Time Progression Demo ===")
    
    # Create simple network with one UE
    sim = Sim()
    
    # One macro, one micro, one UE
    macro = EnhancedCell(sim, bs_type=BSType.MACRO, xyz=(1000, 1000, 25))
    micro = EnhancedCell(sim, bs_type=BSType.MICRO, xyz=(1200, 1200, 10))
    ue = TrafficAwareUE(sim, xyz=(1100, 1100, 2))
    
    # Attach UE to macro
    ue.attach(macro)
    
    print(f"Demonstrating traffic patterns over 24 hours...")
    print(f"UE Type: {ue.traffic_type}")
    print(f"Time\tTraffic(Mbps)\tLow-Load Period")
    print("-" * 40)
    
    # Simulate 24 hours, checking every 2 hours
    for hour in range(0, 24, 2):
        # Update simulation time
        sim.env._now = hour * 3600  # Convert to seconds
        
        # Update UE traffic
        ue.update_traffic()
        
        is_low_load = ue.is_in_low_load_period()
        
        print(f"{hour:02d}:00\t{ue.current_traffic_rate:.3f}\t\t{is_low_load}")

if __name__ == "__main__":
    # Run basic demo
    run_basic_demo()
    
    # Show traffic progression
    simulate_time_progression()
    
    print(f"\n=== Demo Complete ===")
    print(f"Check the log output for detailed cell status over time.")
    print(f"Key observations:")
    print(f"  - Micro cells sleep during low-load period (22:00-06:00)")  
    print(f"  - Users are reallocated to macro cells or other micros")
    print(f"  - Energy savings achieved without QoS degradation")
    print(f"  - Cells wake up automatically when low-load period ends")