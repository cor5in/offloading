# src/core/hetnet_base.py
"""
Base classes and utilities for HetNet simulation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from AIMM_simulator import Sim
from .enhanced_cell import EnhancedCell, BSType, BSState
from .traffic_aware_ue import TrafficAwareUE

class HetNetSimulation:
    """Base class for Heterogeneous Network simulations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sim = None
        self.cells = []
        self.ues = []
        self.ric = None
        self.loggers = []
        
    def create_simulation(self) -> Sim:
        """Create AIMM simulation instance"""
        params = {
            'fc_GHz': self.config.get('carrier_frequency', 2.4),
            'h_UT': self.config.get('ue_height', 2.0),
            'h_BS': self.config.get('bs_height', 25.0)
        }
        
        self.sim = Sim(params=params)
        return self.sim
        
    def deploy_macro_cells(self, positions: List[Tuple[float, float]]) -> List[EnhancedCell]:
        """Deploy macro base stations"""
        macro_cells = []
        
        for i, (x, y) in enumerate(positions):
            cell = EnhancedCell(
                self.sim,
                bs_type=BSType.MACRO,
                xyz=(x, y, self.config.get('macro_height', 25.0)),
                n_subbands=1,
                verbosity=0
            )
            macro_cells.append(cell)
            self.cells.append(cell)
            
        return macro_cells
        
    def deploy_micro_cells(self, positions: List[Tuple[float, float]]) -> List[EnhancedCell]:
        """Deploy micro base stations"""
        micro_cells = []
        
        for i, (x, y) in enumerate(positions):
            cell = EnhancedCell(
                self.sim,
                bs_type=BSType.MICRO,
                xyz=(x, y, self.config.get('micro_height', 10.0)),
                n_subbands=1,
                verbosity=0
            )
            micro_cells.append(cell)
            self.cells.append(cell)
            
        return micro_cells
        
    def deploy_users(self, positions: List[Tuple[float, float]]) -> List[TrafficAwareUE]:
        """Deploy user equipment"""
        users = []
        
        for i, (x, y) in enumerate(positions):
            ue = TrafficAwareUE(
                self.sim,
                xyz=(x, y, self.config.get('ue_height', 2.0)),
                verbosity=0
            )
            users.append(ue)
            self.ues.append(ue)
            
        return users
        
    def setup_cell_associations(self):
        """Setup initial cell-UE associations"""
        for ue in self.ues:
            best_cell = self.find_best_cell_for_ue(ue)
            if best_cell:
                ue.attach(best_cell)
                
    def find_best_cell_for_ue(self, ue: TrafficAwareUE) -> Optional[EnhancedCell]:
        """Find best serving cell for UE based on SINR"""
        best_cell = None
        best_sinr = -float('inf')
        
        for cell in self.cells:
            if cell.state == BSState.ACTIVE:
                sinr = cell.calculate_sinr(ue.xyz, self.cells)
                if sinr > best_sinr and sinr > self.config.get('sinr_threshold', -6):
                    best_sinr = sinr
                    best_cell = cell
                    
        return best_cell
        
    def setup_neighbor_relationships(self):
        """Setup neighbor relationships between cells"""
        threshold = self.config.get('neighbor_threshold', 3000)
        
        for i, cell1 in enumerate(self.cells):
            for j, cell2 in enumerate(self.cells):
                if i != j:
                    distance = np.linalg.norm(
                        np.array(cell1.xyz[:2]) - np.array(cell2.xyz[:2])
                    )
                    if distance < threshold:
                        cell1.add_neighbor(cell2)
                        
    def get_network_statistics(self) -> Dict:
        """Get current network statistics"""
        stats = {
            'total_cells': len(self.cells),
            'macro_cells': len([c for c in self.cells if c.bs_type == BSType.MACRO]),
            'micro_cells': len([c for c in self.cells if c.bs_type == BSType.MICRO]),
            'active_cells': len([c for c in self.cells if c.state == BSState.ACTIVE]),
            'sleeping_cells': len([c for c in self.cells if c.state == BSState.SLEEP]),
            'total_users': len(self.ues),
            'attached_users': sum(len(c.attached) for c in self.cells),
        }
        
        # Calculate average load
        active_cells = [c for c in self.cells if c.state == BSState.ACTIVE]
        if active_cells:
            stats['average_load'] = sum(c.get_load() for c in active_cells) / len(active_cells)
        else:
            stats['average_load'] = 0
            
        # Calculate total power consumption
        stats['total_power'] = sum(c.get_power_consumption() for c in self.cells)
        
        return stats
        
    def print_network_status(self):
        """Print current network status"""
        stats = self.get_network_statistics()
        
        print(f"\n=== Network Status at t={self.sim.env.now:.1f}s ===")
        print(f"Cells: {stats['macro_cells']} macro + {stats['micro_cells']} micro")
        print(f"States: {stats['active_cells']} active, {stats['sleeping_cells']} sleeping")
        print(f"Users: {stats['attached_users']}/{stats['total_users']} attached")
        print(f"Average Load: {stats['average_load']:.3f}")
        print(f"Total Power: {stats['total_power']:.1f} W")
        
        # Cell details
        for cell in self.cells:
            load = cell.get_load()
            power = cell.get_power_consumption()
            users = len(cell.attached)
            print(f"  {cell.bs_type.value} BS[{cell.i}]: {cell.state.value}, "
                  f"Load={load:.3f}, Power={power:.1f}W, Users={users}")

class NetworkTopology:
    """Network topology generator utilities"""
    
    @staticmethod
    def hexagonal_macro_layout(center: Tuple[float, float], 
                              radius: float, 
                              n_rings: int = 1) -> List[Tuple[float, float]]:
        """Generate hexagonal macro cell layout"""
        positions = [center]  # Center cell
        
        for ring in range(1, n_rings + 1):
            for i in range(6):  # 6 sides of hexagon
                for j in range(ring):
                    # Calculate position on hexagon perimeter
                    angle = (i * 60 + j * 60/ring) * np.pi / 180
                    x = center[0] + ring * radius * np.cos(angle)
                    y = center[1] + ring * radius * np.sin(angle)
                    positions.append((x, y))
                    
        return positions
        
    @staticmethod
    def random_micro_layout(area_bounds: Tuple[float, float, float, float],
                           n_cells: int,
                           min_distance: float = 500,
                           seed: Optional[int] = None) -> List[Tuple[float, float]]:
        """Generate random micro cell positions with minimum distance constraint"""
        if seed is not None:
            np.random.seed(seed)
            
        x_min, y_min, x_max, y_max = area_bounds
        positions = []
        
        max_attempts = n_cells * 100  # Prevent infinite loop
        attempts = 0
        
        while len(positions) < n_cells and attempts < max_attempts:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            # Check minimum distance constraint
            valid = True
            for px, py in positions:
                if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                    valid = False
                    break
                    
            if valid:
                positions.append((x, y))
                
            attempts += 1
            
        if len(positions) < n_cells:
            print(f"Warning: Could only place {len(positions)}/{n_cells} micro cells")
            
        return positions
        
    @staticmethod
    def random_ue_layout(area_bounds: Tuple[float, float, float, float],
                        n_users: int,
                        hotspot_centers: Optional[List[Tuple[float, float]]] = None,
                        hotspot_ratio: float = 0.7,
                        seed: Optional[int] = None) -> List[Tuple[float, float]]:
        """Generate random UE positions with optional hotspots"""
        if seed is not None:
            np.random.seed(seed)
            
        x_min, y_min, x_max, y_max = area_bounds
        positions = []
        
        if hotspot_centers and hotspot_ratio > 0:
            # Place some users in hotspots
            n_hotspot = int(n_users * hotspot_ratio)
            hotspot_radius = min(x_max - x_min, y_max - y_min) * 0.1
            
            for i in range(n_hotspot):
                # Choose random hotspot
                hx, hy = hotspot_centers[np.random.randint(len(hotspot_centers))]
                
                # Random position around hotspot
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.exponential(hotspot_radius / 3)
                
                x = hx + radius * np.cos(angle)
                y = hy + radius * np.sin(angle)
                
                # Clamp to bounds
                x = np.clip(x, x_min, x_max)
                y = np.clip(y, y_min, y_max)
                
                positions.append((x, y))
                
            # Place remaining users uniformly
            for i in range(n_users - n_hotspot):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                positions.append((x, y))
        else:
            # Uniform distribution
            for i in range(n_users):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                positions.append((x, y))
                
        return positions

class NetworkValidator:
    """Network validation utilities"""
    
    @staticmethod
    def validate_coverage(cells: List[EnhancedCell], 
                         ues: List[TrafficAwareUE],
                         sinr_threshold: float = -6.0) -> Dict:
        """Validate network coverage"""
        results = {
            'total_ues': len(ues),
            'covered_ues': 0,
            'uncovered_ues': 0,
            'coverage_ratio': 0.0,
            'uncovered_positions': []
        }
        
        for ue in ues:
            best_sinr = -float('inf')
            
            for cell in cells:
                if cell.state == BSState.ACTIVE:
                    sinr = cell.calculate_sinr(ue.xyz, cells)
                    best_sinr = max(best_sinr, sinr)
                    
            if best_sinr >= sinr_threshold:
                results['covered_ues'] += 1
            else:
                results['uncovered_ues'] += 1
                results['uncovered_positions'].append(ue.xyz[:2])
                
        results['coverage_ratio'] = results['covered_ues'] / results['total_ues']
        
        return results
        
    @staticmethod
    def validate_capacity(cells: List[EnhancedCell], 
                         ues: List[TrafficAwareUE]) -> Dict:
        """Validate network capacity"""
        results = {
            'overloaded_cells': 0,
            'underloaded_cells': 0,
            'balanced_cells': 0,
            'max_load': 0.0,
            'avg_load': 0.0
        }
        
        loads = []
        for cell in cells:
            if cell.state == BSState.ACTIVE:
                load = cell.get_load()
                loads.append(load)
                
                if load > 0.9:
                    results['overloaded_cells'] += 1
                elif load < 0.1:
                    results['underloaded_cells'] += 1
                else:
                    results['balanced_cells'] += 1
                    
        if loads:
            results['max_load'] = max(loads)
            results['avg_load'] = sum(loads) / len(loads)
            
        return results