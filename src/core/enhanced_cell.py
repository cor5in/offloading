# src/core/enhanced_cell.py
"""
Enhanced Cell implementation for Heterogeneous Cellular Networks
Based on MiLSF paper implementation
"""

import numpy as np
from math import log10, log2
from enum import Enum
from collections import deque
from AIMM_simulator import Cell

class BSType(Enum):
    MACRO = "macro"
    MICRO = "micro"

class BSState(Enum):
    ACTIVE = "active"
    SLEEP = "sleep"

class EnhancedCell(Cell):
    """Enhanced Cell class with heterogeneous network capabilities"""
    
    def __init__(self, sim, bs_type=BSType.MACRO, **kwargs):
        super().__init__(sim, **kwargs)
        
        self.bs_type = bs_type
        self.state = BSState.ACTIVE
        self.neighbor_cells = set()
        self.traffic_history = deque(maxlen=168)  # One week of hourly data
        
        # Set parameters based on BS type (from Table II)
        self._configure_bs_parameters()
        
        # Convert dBm to linear scale for calculations
        self.tx_power_linear = self.from_dBm(self.power_dBm)
        
    def _configure_bs_parameters(self):
        """Configure BS parameters based on type"""
        if self.bs_type == BSType.MACRO:
            # Macro BS parameters from Table II
            self.power_dBm = 46.0  # 8W transmit power per antenna
            self.n_antennas = 6
            self.carrier_freq = 2.4e9  # 2.4 GHz
            self.bandwidth = 20e6  # 20 MHz
            self.circuit_power = 120.0  # Watts
            self.sleep_power = 8.0  # Watts
            self.coverage_radius = 2700.0  # meters
        else:  # MICRO
            # Micro BS parameters from Table II
            self.power_dBm = 38.0  # 3W transmit power per antenna
            self.n_antennas = 2
            self.carrier_freq = 20e9  # 20 GHz
            self.bandwidth = 50e6  # 50 MHz
            self.circuit_power = 10.0  # Watts
            self.sleep_power = 2.0  # Watts
            self.coverage_radius = 1000.0  # meters
            
    def from_dBm(self, power_dBm):
        """Convert power from dBm to linear scale"""
        return 10**((power_dBm - 30) / 10)
        
    def to_dB(self, linear_value):
        """Convert linear value to dB"""
        return 10 * log10(max(linear_value, 1e-12))
        
    def calculate_pathloss(self, distance):
        """Calculate pathloss using 3GPP Urban Macro model"""
        fc_GHz = self.carrier_freq / 1e9
        
        if distance < 10:
            distance = 10  # Minimum distance
            
        # 3GPP UMa LOS model (Equation 2 from paper)
        PL_LOS = 28.0 + 20*log10(fc_GHz) + 22*log10(distance)
        
        return PL_LOS
        
    def calculate_sinr(self, ue_position, interfering_cells):
        """Calculate SINR for UE at given position (Equation 3)"""
        distance = np.linalg.norm(self.xyz[:2] - ue_position[:2])
        
        # Signal power
        pathloss_dB = self.calculate_pathloss(distance)
        signal_power = self.tx_power_linear * (10**(-pathloss_dB/10))
        
        # Interference power
        interference_power = 0
        for cell in interfering_cells:
            if cell != self and hasattr(cell, 'state') and cell.state == BSState.ACTIVE:
                int_distance = np.linalg.norm(cell.xyz[:2] - ue_position[:2])
                int_pathloss_dB = cell.calculate_pathloss(int_distance)
                interference_power += cell.tx_power_linear * (10**(-int_pathloss_dB/10))
        
        # Noise power (thermal noise) - η₀w from paper
        noise_power = 1e-13  # -100 dBm
        
        # SINR calculation
        sinr = signal_power / (interference_power + noise_power)
        return self.to_dB(sinr)
        
    def calculate_required_bandwidth(self, traffic_rate, sinr_dB):
        """Calculate required bandwidth using Shannon-Hartley theorem (Equation 4)"""
        sinr_linear = 10**(sinr_dB/10)
        required_bw = traffic_rate / log2(1 + sinr_linear)
        return required_bw
        
    def get_load(self):
        """Calculate current cell load (Equation 5)"""
        total_required_bw = 0
        
        for ue_id in self.attached:
            ue = self.sim.UEs[ue_id]
            sinr_dB = self.calculate_sinr(ue.xyz, self.sim.cells)
            
            if sinr_dB > -6:  # SINR threshold γ₀ = -6 dB
                traffic_rate = getattr(ue, 'current_traffic_rate', 1.0)  # Mbps
                required_bw = self.calculate_required_bandwidth(traffic_rate, sinr_dB)
                total_required_bw += required_bw
                
        load = total_required_bw / self.bandwidth
        return min(load, 1.0)
        
    def get_power_consumption(self):
        """Calculate power consumption based on current state and load (Equation 6)"""
        if self.state == BSState.SLEEP:
            return self.sleep_power
        else:
            load = self.get_load()
            # Linear model: P = α * p * μ + p_c
            transmit_power = self.n_antennas * self.from_dBm(self.power_dBm) * load
            return transmit_power + self.circuit_power
            
    def switch_to_sleep(self):
        """Switch cell to sleep mode"""
        if self.bs_type == BSType.MICRO:  # Only micro cells can sleep
            self.state = BSState.SLEEP
            print(f"MiBS[{self.i}] switched to SLEEP mode at t={self.sim.env.now:.1f}")
            
    def switch_to_active(self):
        """Switch cell to active mode"""
        self.state = BSState.ACTIVE
        print(f"BS[{self.i}] switched to ACTIVE mode at t={self.sim.env.now:.1f}")
        
    def add_neighbor(self, cell):
        """Add neighboring cell"""
        self.neighbor_cells.add(cell.i)
        
    def can_serve_ue(self, ue, max_traffic_rate):
        """Check if this cell can serve the given UE with its maximum traffic rate"""
        if self.state != BSState.ACTIVE:
            return False
            
        # Check SINR threshold
        sinr_dB = self.calculate_sinr(ue.xyz, self.sim.cells)
        if sinr_dB < -6:  # γ₀ = -6 dB
            return False
            
        # Check capacity constraint
        required_bw = self.calculate_required_bandwidth(max_traffic_rate, sinr_dB)
        current_load = self.get_load()
        additional_load = required_bw / self.bandwidth
        
        return current_load + additional_load <= 1.0
        
    def get_serving_quality(self, ue):
        """Get the serving quality (SINR) for a UE"""
        return self.calculate_sinr(ue.xyz, self.sim.cells)