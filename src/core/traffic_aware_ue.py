# src/core/traffic_aware_ue.py
"""
Traffic-aware UE implementation with realistic traffic patterns
Based on MiLSF paper traffic modeling
"""

import numpy as np
from collections import deque
from AIMM_simulator import UE

class TrafficAwareUE(UE):
    """UE with traffic prediction capabilities and realistic traffic patterns"""
    
    def __init__(self, sim, **kwargs):
        super().__init__(sim, **kwargs)
        
        self.traffic_history = deque(maxlen=168)  # One week of hourly data
        self.current_traffic_rate = 1.0  # Mbps
        self.max_traffic_rate = 0.0
        self.base_traffic = np.random.uniform(0.5, 2.0)  # Individual base traffic
        
        # Traffic pattern parameters
        self.traffic_type = np.random.choice(['business', 'residential', 'mixed'], 
                                           p=[0.3, 0.4, 0.3])
        
        # Generate realistic traffic pattern
        self.generate_initial_traffic_pattern()
        
    def generate_initial_traffic_pattern(self):
        """Generate realistic daily traffic pattern based on user type"""
        # Simulate one week of hourly traffic data
        for hour in range(168):  # 7 days * 24 hours
            daily_hour = hour % 24
            day_of_week = hour // 24
            
            # Generate traffic based on user type and time
            traffic = self._calculate_traffic_for_time(daily_hour, day_of_week)
            self.traffic_history.append(traffic)
            
        self.max_traffic_rate = max(self.traffic_history)
        self.current_traffic_rate = self.traffic_history[-1]
        
    def _calculate_traffic_for_time(self, hour, day_of_week):
        """Calculate traffic rate for specific time"""
        base_multiplier = 1.0
        
        # Business user pattern
        if self.traffic_type == 'business':
            if 0 <= day_of_week <= 4:  # Weekdays
                if 9 <= hour <= 17:  # Business hours
                    base_multiplier = 2.0 + 0.5 * np.sin((hour - 9) * np.pi / 8)
                elif 8 <= hour <= 8 or 18 <= hour <= 19:  # Commute hours
                    base_multiplier = 1.5
                else:  # Off hours
                    base_multiplier = 0.3
            else:  # Weekends
                base_multiplier = 0.5 + 0.3 * np.random.random()
                
        # Residential user pattern  
        elif self.traffic_type == 'residential':
            if 6 <= hour <= 22:  # Active hours
                # Peak in evening
                if 19 <= hour <= 22:
                    base_multiplier = 1.5 + 0.8 * np.sin((hour - 19) * np.pi / 3)
                # Morning peak
                elif 7 <= hour <= 9:
                    base_multiplier = 1.2 + 0.3 * np.sin((hour - 7) * np.pi / 2)
                else:
                    base_multiplier = 0.8 + 0.2 * np.sin((hour - 6) * np.pi / 16)
            else:  # Night hours (low-load period)
                base_multiplier = 0.1 + 0.1 * np.random.random()
                
        # Mixed user pattern
        else:  # mixed
            if 6 <= hour <= 22:
                base_multiplier = 0.8 + 0.6 * np.sin((hour - 6) * np.pi / 16)
            else:
                base_multiplier = 0.2 + 0.1 * np.random.random()
                
        # Add day-of-week variation
        if day_of_week >= 5:  # Weekend
            base_multiplier *= 0.7
            
        # Add random noise
        noise = 0.1 * np.random.normal()
        traffic = self.base_traffic * (base_multiplier + noise)
        
        return max(0.1, traffic)  # Minimum traffic
        
    def update_traffic(self):
        """Update current traffic based on time of day"""
        current_hour = int(self.sim.env.now / 3600) % 24  # Convert sim time to hours
        current_day = int(self.sim.env.now / 86400) % 7   # Day of week
        
        # Calculate new traffic rate
        self.current_traffic_rate = self._calculate_traffic_for_time(current_hour, current_day)
        
        # Update history
        self.traffic_history.append(self.current_traffic_rate)
        
        # Update maximum (for constraint checking)
        self.max_traffic_rate = max(self.traffic_history)
        
    def get_traffic_history(self, hours=24):
        """Get recent traffic history"""
        if len(self.traffic_history) >= hours:
            return list(self.traffic_history)[-hours:]
        else:
            return list(self.traffic_history)
            
    def predict_peak_traffic(self, future_hours=8):
        """Predict maximum traffic in the next future_hours"""
        current_hour = int(self.sim.env.now / 3600) % 24
        current_day = int(self.sim.env.now / 86400) % 7
        
        peak_traffic = 0
        for h in range(future_hours):
            future_hour = (current_hour + h) % 24
            future_day = (current_day + (current_hour + h) // 24) % 7
            traffic = self._calculate_traffic_for_time(future_hour, future_day)
            peak_traffic = max(peak_traffic, traffic)
            
        return peak_traffic
        
    def is_in_low_load_period(self):
        """Check if UE is currently in low-load period"""
        current_hour = int(self.sim.env.now / 3600) % 24
        
        # Low-load period: 22:00 - 06:00 (night time)
        if 22 <= current_hour or current_hour < 6:
            return True
        return False
        
    def get_qos_requirement(self):
        """Get QoS requirement based on current traffic"""
        # Higher traffic requires better QoS
        if self.current_traffic_rate > 2.0:
            return -3  # Higher SINR requirement
        elif self.current_traffic_rate > 1.0:
            return -6  # Standard SINR requirement
        else:
            return -9  # Lower SINR acceptable for low traffic
            
    # Override attach method to be compatible with EnhancedCell
    def attach(self, cell):
        """Attach this UE to a specific Cell instance."""
        if hasattr(cell, 'attached'):
            cell.attached.add(self.i)
        self.serving_cell = cell
        
        # Set serving cell ids for handover tracking
        if hasattr(self, 'serving_cell_ids'):
            self.serving_cell_ids.appendleft((cell.i, self.sim.env.now))
        
        if self.verbosity > 0:
            print(f'UE[{self.i:2}] attached to {cell.bs_type.value if hasattr(cell, "bs_type") else ""}cell[{cell.i}]')
    
    def detach(self, quiet=True):
        """Detach this UE from its serving cell."""
        if self.serving_cell is None:
            return
            
        if hasattr(self.serving_cell, 'attached'):
            self.serving_cell.attached.discard(self.i)
            
        # Clear saved reports from this UE
        if hasattr(self.serving_cell, 'reports'):
            reports = self.serving_cell.reports
            for x in reports:
                if self.i in reports[x]: 
                    del reports[x][self.i]
        
        if not quiet and self.verbosity > 0:
            print(f'UE[{self.i}] detached from cell[{self.serving_cell.i}]')
            
        self.serving_cell = None
        
    def attach_to_strongest_cell_simple_pathloss_model(self):
        """
        Attach to the cell delivering the strongest signal
        at the current UE position. Compatible with enhanced cells.
        """
        best_cell = None
        best_sinr = -float('inf')
        
        for cell in self.sim.cells:
            if hasattr(cell, 'state') and hasattr(cell, 'calculate_sinr'):
                # Use enhanced cell's SINR calculation if available
                if cell.state.value == 'active':  # Check if cell is active
                    sinr = cell.calculate_sinr(self.xyz, self.sim.cells)
                    if sinr > best_sinr and sinr > -6:  # SINR threshold
                        best_sinr = sinr
                        best_cell = cell
            else:
                # Fallback to simple distance-based selection
                distance = np.linalg.norm(np.array(cell.xyz) - np.array(self.xyz))
                signal_strength = -20 * np.log10(distance + 1)  # Simple path loss
                if signal_strength > best_sinr:
                    best_sinr = signal_strength
                    best_cell = cell
        
        if best_cell:
            self.attach(best_cell)
            if self.verbosity > 0:
                cell_type = best_cell.bs_type.value if hasattr(best_cell, 'bs_type') else 'unknown'
                print(f'UE[{self.i:2}] ⟵⟶ {cell_type} cell[{best_cell.i}] (SINR: {best_sinr:.1f} dB)')
        else:
            if self.verbosity > 0:
                print(f'UE[{self.i}] could not attach to any cell')
                
    def attach_to_nearest_cell(self):
        """
        Attach this UE to the geographically nearest Cell instance.
        Enhanced version compatible with new cell types.
        """
        best_cell = None
        min_distance = float('inf')
        
        for cell in self.sim.cells:
            distance = np.linalg.norm(np.array(cell.xyz[:2]) - np.array(self.xyz[:2]))
            if distance < min_distance:
                min_distance = distance
                best_cell = cell
                
        if best_cell:
            self.attach(best_cell)
            if self.verbosity > 0:
                cell_type = best_cell.bs_type.value if hasattr(best_cell, 'bs_type') else 'unknown'
                print(f'UE[{self.i:2}] ⟵⟶ {cell_type} cell[{best_cell.i}] (distance: {min_distance:.1f}m)')
        else:
            if self.verbosity > 0:
                print(f'UE[{self.i}] could not attach to any cell')
                
    def get_serving_cell(self):
        """Return the current serving Cell object (not index) for this UE instance."""
        return self.serving_cell
        
    def get_serving_cell_i(self):
        """Return the current serving Cell index for this UE instance."""
        if self.serving_cell is None:
            return None
        return self.serving_cell.i
        
    def send_rsrp_reports(self, threshold=-120.0):
        """
        Send RSRP reports in dBm to all cells for which it is over the threshold.
        Enhanced version compatible with EnhancedCell.
        """
        for cell in self.sim.cells:
            if hasattr(cell, 'calculate_sinr'):
                # Use enhanced cell's calculation
                sinr_dB = cell.calculate_sinr(self.xyz, self.sim.cells)
                # Convert SINR to approximate RSRP (simplified)
                rsrp_dBm = sinr_dB - 10  # Rough approximation
            else:
                # Fallback to simple path loss calculation
                distance = np.linalg.norm(np.array(cell.xyz) - np.array(self.xyz))
                rsrp_dBm = cell.power_dBm - 20 * np.log10(distance + 1)
            
            if rsrp_dBm > threshold:
                if hasattr(cell, 'reports'):
                    cell.reports['rsrp'][self.i] = (self.sim.env.now, rsrp_dBm)
                    
                    if self.i not in cell.rsrp_history:
                        cell.rsrp_history[self.i] = deque([-np.inf,]*10, maxlen=10)
                    cell.rsrp_history[self.i].appendleft(rsrp_dBm)
                    
    def send_subband_cqi_report(self):
        """
        Enhanced CQI reporting compatible with EnhancedCell.
        """
        if self.serving_cell is None:
            return 0.0
            
        if hasattr(self.serving_cell, 'calculate_sinr'):
            # Use enhanced cell's SINR calculation
            sinr_dB = self.serving_cell.calculate_sinr(self.xyz, self.sim.cells)
        else:
            # Fallback calculation
            distance = np.linalg.norm(np.array(self.serving_cell.xyz) - np.array(self.xyz))
            sinr_dB = self.serving_cell.power_dBm - 20 * np.log10(distance + 1) - (-100)  # Simplified
        
        # Convert SINR to CQI (simplified mapping)
        if sinr_dB >= 20:
            cqi = 15
        elif sinr_dB >= -6:
            cqi = max(1, min(15, int((sinr_dB + 6) / 26 * 14) + 1))
        else:
            cqi = 0
            
        # Calculate throughput (simplified Shannon formula)
        if sinr_dB > -10:
            sinr_linear = 10**(sinr_dB/10)
            spectral_efficiency = np.log2(1 + sinr_linear)
            throughput_Mbps = self.serving_cell.bandwidth / 1e6 * spectral_efficiency
            
            # Adjust for current traffic and cell load
            throughput_Mbps *= min(1.0, self.current_traffic_rate / 5.0)  # Traffic scaling
            if hasattr(self.serving_cell, 'get_load'):
                load = self.serving_cell.get_load()
                throughput_Mbps *= (1 - load * 0.5)  # Load impact
        else:
            throughput_Mbps = 0.1  # Minimum throughput
            
        # Store in cell reports
        if hasattr(self.serving_cell, 'reports'):
            now = float(self.sim.env.now)
            self.serving_cell.reports['cqi'][self.i] = (now, [cqi])
            self.serving_cell.reports['throughput_Mbps'][self.i] = (now, throughput_Mbps)
            
        # Store in UE
        self.cqi = [cqi]
        self.sinr_dB = sinr_dB
        
        return throughput_Mbps