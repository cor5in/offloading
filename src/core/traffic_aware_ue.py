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