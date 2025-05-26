# src/algorithms/milsf_ric.py
"""
MiLSF (Minimum Load Sleep First) RIC Implementation
Based on Algorithm 1 from the paper
"""

from AIMM_simulator import RIC
from ..core.enhanced_cell import BSType, BSState
import sys

class MiLSF_RIC(RIC):
    """MiLSF (Minimum Load Sleep First) RIC Implementation"""
    
    def __init__(self, sim, interval=30.0, low_load_start=22, low_load_end=6, 
                 sinr_threshold=-6.0, verbosity=1):
        super().__init__(sim, interval, verbosity)
        
        self.low_load_start = low_load_start  # 10:00 PM
        self.low_load_end = low_load_end      # 6:00 AM
        self.sinr_threshold = sinr_threshold  # γ₀ = -6 dB
        self.sleeping_cells = set()
        
        # Energy tracking
        self.total_energy_without_sleeping = 0.0
        self.total_energy_with_sleeping = 0.0
        self.energy_savings_history = []
        
        # Statistics
        self.sleep_decisions = 0
        self.wake_decisions = 0
        self.failed_reallocations = 0
        
    def is_low_load_period(self):
        """Check if current time is in low-load period"""
        current_hour = int(self.sim.env.now / 3600) % 24  # Convert to hours
        
        if self.low_load_start > self.low_load_end:  # Crosses midnight
            return current_hour >= self.low_load_start or current_hour < self.low_load_end
        else:
            return self.low_load_start <= current_hour < self.low_load_end
            
    def get_micro_cells(self):
        """Get all micro base stations"""
        return [cell for cell in self.sim.cells 
                if hasattr(cell, 'bs_type') and cell.bs_type == BSType.MICRO]
        
    def get_macro_cells(self):
        """Get all macro base stations"""
        return [cell for cell in self.sim.cells 
                if hasattr(cell, 'bs_type') and cell.bs_type == BSType.MACRO]
        
    def get_active_cells(self):
        """Get all currently active cells"""
        return [cell for cell in self.sim.cells 
                if hasattr(cell, 'state') and cell.state == BSState.ACTIVE]
        
    def can_reallocate_users(self, target_cell, candidate_cells):
        """Check if all users of target_cell can be reallocated (Step 2 in Algorithm 1)"""
        users_to_reallocate = list(target_cell.attached)
        
        if not users_to_reallocate:  # No users to reallocate
            return True
            
        # For each user, check if at least one candidate cell can serve it
        for ue_id in users_to_reallocate:
            ue = self.sim.UEs[ue_id]
            can_serve = False
            
            # Get maximum traffic rate during low-load period
            max_traffic_rate = getattr(ue, 'max_traffic_rate', ue.current_traffic_rate)
            
            for candidate_cell in candidate_cells:
                if candidate_cell.can_serve_ue(ue, max_traffic_rate):
                    can_serve = True
                    break
                    
            if not can_serve:
                return False
                
        return True
        
    def reallocate_users(self, source_cell, candidate_cells):
        """Reallocate users from source_cell using MiLSF strategy (Step 3 in Algorithm 1)"""
        users_to_reallocate = list(source_cell.attached)
        successful_reallocations = 0
        
        for ue_id in users_to_reallocate:
            ue = self.sim.UEs[ue_id]
            best_cell = None
            
            # Step 1: Try to allocate to MaBS with highest SINR (Lines 6-15)
            macro_candidates = [cell for cell in candidate_cells if cell.bs_type == BSType.MACRO]
            best_sinr = -float('inf')
            
            for macro_cell in macro_candidates:
                if macro_cell.can_serve_ue(ue, ue.max_traffic_rate):
                    sinr = macro_cell.get_serving_quality(ue)
                    if sinr > best_sinr:
                        best_sinr = sinr
                        best_cell = macro_cell
                        
            # Step 2: If no MaBS available, try MiBS with highest load (Lines 17-28)
            if best_cell is None:
                micro_candidates = [cell for cell in candidate_cells if cell.bs_type == BSType.MICRO]
                best_load = -1
                
                for micro_cell in micro_candidates:
                    if micro_cell.can_serve_ue(ue, ue.max_traffic_rate):
                        load = micro_cell.get_load()
                        if load > best_load:
                            best_load = load
                            best_cell = micro_cell
                            
            # Perform reallocation
            if best_cell is not None:
                source_cell.attached.remove(ue_id)
                best_cell.attached.add(ue_id)
                ue.serving_cell = best_cell
                successful_reallocations += 1
                
                if self.verbosity > 1:
                    print(f"  UE[{ue_id}] reallocated from {source_cell.bs_type.value}[{source_cell.i}] "
                          f"to {best_cell.bs_type.value}[{best_cell.i}]")
                          
        return successful_reallocations == len(users_to_reallocate)
        
    def milsf_algorithm(self):
        """Implement MiLSF algorithm (Algorithm 1)"""
        if not self.is_low_load_period():
            return []
            
        micro_cells = self.get_micro_cells()
        active_micro_cells = [cell for cell in micro_cells if cell.state == BSState.ACTIVE]
        
        # Step 1: Sort MiBSs by load (ascending order)
        active_micro_cells.sort(key=lambda cell: cell.get_load())
        
        decisions = []
        
        # Step 2: Try to sleep each MiBS starting from lowest load
        for cell in active_micro_cells:
            # Get active cells (both MaBSs and MiBSs) as candidates
            candidate_cells = [c for c in self.get_active_cells() if c != cell]
            
            # Step 3: Check if users can be reallocated (Lines 5-33)
            if self.can_reallocate_users(cell, candidate_cells):
                # Step 4: Reallocate users
                if self.reallocate_users(cell, candidate_cells):
                    # Step 5: Switch cell to sleep (Lines 34-35)
                    load_before_sleep = cell.get_load()
                    cell.switch_to_sleep()
                    self.sleeping_cells.add(cell.i)
                    self.sleep_decisions += 1
                    
                    decision_msg = f"MiBS[{cell.i}] sleeping (load={load_before_sleep:.3f})"
                    decisions.append(decision_msg)
                    
                    if self.verbosity > 0:
                        print(f"t={self.sim.env.now:.1f}: {decision_msg}")
                else:
                    self.failed_reallocations += 1
                    if self.verbosity > 1:
                        print(f"t={self.sim.env.now:.1f}: Failed to reallocate users from MiBS[{cell.i}]")
                        
        return decisions
        
    def wake_up_cells(self):
        """Wake up all sleeping cells when exiting low-load period"""
        if self.sleeping_cells:
            decisions = []
            for cell_id in list(self.sleeping_cells):
                cell = self.sim.cells[cell_id]
                if cell.state == BSState.SLEEP:
                    cell.switch_to_active()
                    self.wake_decisions += 1
                    decisions.append(f"MiBS[{cell_id}] activated")
                    
            self.sleeping_cells.clear()
            
            if self.verbosity > 0:
                print(f"t={self.sim.env.now:.1f}: All sleeping cells activated")
                
            return decisions
        return []
        
    def calculate_energy_savings(self):
        """Calculate current energy savings percentage"""
        total_energy_active = 0
        total_energy_current = 0
        
        for cell in self.sim.cells:
            if hasattr(cell, 'get_power_consumption'):
                # Current power consumption
                current_power = cell.get_power_consumption()
                total_energy_current += current_power
                
                # Power if all cells were active
                if cell.state == BSState.SLEEP:
                    # Calculate what power would be if active
                    original_state = cell.state
                    cell.state = BSState.ACTIVE
                    active_power = cell.get_power_consumption()
                    cell.state = original_state
                    total_energy_active += active_power
                else:
                    total_energy_active += current_power
                    
        # Update cumulative energy
        self.total_energy_without_sleeping += total_energy_active * self.interval
        self.total_energy_with_sleeping += total_energy_current * self.interval
        
        # Calculate instantaneous savings
        if total_energy_active > 0:
            savings_percentage = ((total_energy_active - total_energy_current) / total_energy_active) * 100
            self.energy_savings_history.append(savings_percentage)
            return savings_percentage
        return 0
        
    def get_network_statistics(self):
        """Get current network statistics"""
        active_micros = len([c for c in self.get_micro_cells() if c.state == BSState.ACTIVE])
        sleeping_micros = len([c for c in self.get_micro_cells() if c.state == BSState.SLEEP])
        
        total_load = sum(cell.get_load() for cell in self.get_active_cells())
        avg_load = total_load / len(self.get_active_cells()) if self.get_active_cells() else 0
        
        return {
            'active_micros': active_micros,
            'sleeping_micros': sleeping_micros,
            'total_cells': len(self.sim.cells),
            'average_load': avg_load,
            'low_load_period': self.is_low_load_period()
        }
        
    def loop(self):
        """Main RIC control loop"""
        if self.verbosity > 0:
            print(f"MiLSF RIC started at t={self.sim.env.now:.1f}")
            
        while True:
            # Update UE traffic patterns
            for ue in self.sim.UEs:
                if hasattr(ue, 'update_traffic'):
                    ue.update_traffic()
                    
            # Execute MiLSF algorithm or wake up cells
            if self.is_low_load_period():
                decisions = self.milsf_algorithm()
            else:
                decisions = self.wake_up_cells()
                
            # Calculate energy savings
            energy_savings_pct = self.calculate_energy_savings()
            
            # Log decisions and metrics
            if decisions and self.verbosity > 0:
                print(f"t={self.sim.env.now:.1f} MiLSF Decisions: {'; '.join(decisions)}")
                
            if self.verbosity > 0:
                stats = self.get_network_statistics()
                print(f"t={self.sim.env.now:.1f} Stats: Active MiBS: {stats['active_micros']}, "
                      f"Sleeping: {stats['sleeping_micros']}, Energy savings: {energy_savings_pct:.2f}%")
                      
            yield self.sim.wait(self.interval)
            
    def finalize(self):
        """Print final statistics"""
        if self.total_energy_without_sleeping > 0:
            total_savings = ((self.total_energy_without_sleeping - self.total_energy_with_sleeping) 
                           / self.total_energy_without_sleeping) * 100
                           
            avg_savings = sum(self.energy_savings_history) / len(self.energy_savings_history) if self.energy_savings_history else 0
            
            print(f"\n=== MiLSF Final Statistics ===")
            print(f"Total Energy Savings: {total_savings:.2f}%")
            print(f"Average Energy Savings: {avg_savings:.2f}%")
            print(f"Sleep Decisions: {self.sleep_decisions}")
            print(f"Wake Decisions: {self.wake_decisions}")
            print(f"Failed Reallocations: {self.failed_reallocations}")
            print(f"Total Energy without Sleeping: {self.total_energy_without_sleeping:.2f} Wh")
            print(f"Total Energy with MiLSF: {self.total_energy_with_sleeping:.2f} Wh")
            print("================================")