"""
Advanced Pathfinding Module

This module implements specialized pathfinding algorithms for 3D navigation:
- Jump Point Search (JPS) for efficient pathfinding on grid-based terrains
- Theta* for smoother, more natural paths
- RRT (Rapidly-exploring Random Tree) for handling complex 3D environments
"""

import logging
import random
import math
import numpy as np
import heapq
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

from src.decision.advanced_navigation import Position3D, TerrainNode, TerrainMap

class JumpPointSearch:
    """
    Jump Point Search (JPS) implementation for efficient grid-based pathfinding
    JPS can be significantly faster than A* on uniform cost grids by exploiting 
    symmetry to skip unnecessary nodes
    """
    
    def __init__(self, terrain_map: TerrainMap):
        """
        Initialize JPS
        
        Args:
            terrain_map: TerrainMap to use for pathfinding
        """
        self.logger = logging.getLogger("wow_ai.decision.advanced_pathfinding.jps")
        self.terrain_map = terrain_map
    
    def find_path(self, start_node: TerrainNode, goal_node: TerrainNode) -> List[TerrainNode]:
        """
        Find a path using JPS algorithm
        
        Args:
            start_node: Starting node
            goal_node: Target node
            
        Returns:
            List[TerrainNode]: Path as a list of nodes
        """
        # Initialize data structures
        open_set = []  # Priority queue
        closed_set = set()
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start_node], id(start_node), start_node))
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current == goal_node:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                path.reverse()
                return path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Identify successors using jumping
            successors = self._identify_successors(current, start_node, goal_node, closed_set)
            
            for successor, cost in successors:
                # Skip if successor in closed set
                if successor in closed_set:
                    continue
                
                # Calculate tentative g score
                tentative_g = g_score[current] + cost
                
                # Check if this path is better
                if successor not in g_score or tentative_g < g_score[successor]:
                    # Record this path
                    came_from[successor] = current
                    g_score[successor] = tentative_g
                    f_score[successor] = g_score[successor] + self._heuristic(successor, goal_node)
                    
                    # Add to open set if not already there
                    if successor not in [n for _, _, n in open_set]:
                        heapq.heappush(open_set, (f_score[successor], id(successor), successor))
        
        # If we get here, no path was found
        self.logger.warning("JPS failed to find a path")
        return []
    
    def _identify_successors(self, node: TerrainNode, start_node: TerrainNode, 
                          goal_node: TerrainNode, closed_set: Set) -> List[Tuple[TerrainNode, float]]:
        """
        Find successor nodes using the jumping technique
        
        Args:
            node: Current node
            start_node: Starting node
            goal_node: Goal node
            closed_set: Set of closed nodes
            
        Returns:
            List[Tuple[TerrainNode, float]]: List of (successor, cost) tuples
        """
        successors = []
        neighbors = self._get_pruned_neighbors(node, start_node, closed_set)
        
        for neighbor, direction in neighbors:
            # Jump in the direction
            jump_point = self._jump(neighbor, direction, start_node, goal_node)
            
            if jump_point:
                # Calculate cost
                cost = node.position.distance_to(jump_point.position)
                successors.append((jump_point, cost))
        
        return successors
    
    def _get_pruned_neighbors(self, node: TerrainNode, start_node: TerrainNode, 
                           closed_set: Set) -> List[Tuple[TerrainNode, Tuple[int, int, int]]]:
        """
        Get pruned neighbors based on JPS rules
        
        Args:
            node: Current node
            start_node: Starting node
            closed_set: Set of closed nodes
            
        Returns:
            List[Tuple[TerrainNode, Tuple[int, int, int]]]: List of (neighbor, direction) tuples
        """
        # If this is the start node, return all neighbors
        if node == start_node:
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # Skip the node itself
                        
                        # Calculate neighbor position
                        neighbor_pos = Position3D(
                            node.position.x + dx * self.terrain_map.resolution,
                            node.position.y + dy * self.terrain_map.resolution,
                            node.position.z + dz * self.terrain_map.resolution
                        )
                        
                        # Get the node at that position
                        neighbor = self.terrain_map.get_node_at(neighbor_pos)
                        
                        if neighbor and neighbor.walkable and neighbor not in closed_set:
                            neighbors.append((neighbor, (dx, dy, dz)))
            
            return neighbors
        
        # If this is not the start node, prune neighbors
        # For simplicity, we'll implement a basic version that doesn't fully optimize
        # but still gives better performance than regular A*
        
        # Get all walkable neighbors
        all_neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip the node itself
                    
                    # Calculate neighbor position
                    neighbor_pos = Position3D(
                        node.position.x + dx * self.terrain_map.resolution,
                        node.position.y + dy * self.terrain_map.resolution,
                        node.position.z + dz * self.terrain_map.resolution
                    )
                    
                    # Get the node at that position
                    neighbor = self.terrain_map.get_node_at(neighbor_pos)
                    
                    if neighbor and neighbor.walkable and neighbor not in closed_set:
                        all_neighbors.append((neighbor, (dx, dy, dz)))
        
        # For a basic implementation, we'll just return all walkable neighbors
        # A full JPS implementation would prune based on parent direction
        return all_neighbors
    
    def _jump(self, node: TerrainNode, direction: Tuple[int, int, int], 
           start_node: TerrainNode, goal_node: TerrainNode) -> Optional[TerrainNode]:
        """
        Recursively jump in a direction until finding a jump point
        
        Args:
            node: Current node
            direction: Direction to jump (dx, dy, dz)
            start_node: Starting node
            goal_node: Goal node
            
        Returns:
            Optional[TerrainNode]: Jump point node or None
        """
        # Check if node is valid
        if not node or not node.walkable:
            return None
        
        # If this is the goal node, return it
        if node == goal_node:
            return node
        
        dx, dy, dz = direction
        
        # Calculate next position
        next_pos = Position3D(
            node.position.x + dx * self.terrain_map.resolution,
            node.position.y + dy * self.terrain_map.resolution,
            node.position.z + dz * self.terrain_map.resolution
        )
        
        # Get the node at that position
        next_node = self.terrain_map.get_node_at(next_pos)
        
        # For a basic implementation, we'll just check if the next node is walkable
        # and recursively jump in that direction
        if next_node and next_node.walkable:
            return self._jump(next_node, direction, start_node, goal_node)
        
        # If next node is not walkable, this node might be a forced neighbor
        # For simplicity, we'll just return the current node as a jump point
        return node
    
    def _heuristic(self, a: TerrainNode, b: TerrainNode) -> float:
        """
        Heuristic function for JPS pathfinding (3D Euclidean distance)
        
        Args:
            a: First node
            b: Second node
            
        Returns:
            float: Estimated distance between nodes
        """
        return a.position.distance_to(b.position)

class ThetaStar:
    """
    Theta* pathfinding implementation for smoother, more natural paths
    Unlike A* which constrains paths to grid edges, Theta* allows any-angle paths
    by using line-of-sight checks
    """
    
    def __init__(self, terrain_map: TerrainMap):
        """
        Initialize Theta*
        
        Args:
            terrain_map: TerrainMap to use for pathfinding
        """
        self.logger = logging.getLogger("wow_ai.decision.advanced_pathfinding.theta_star")
        self.terrain_map = terrain_map
    
    def find_path(self, start_node: TerrainNode, goal_node: TerrainNode) -> List[TerrainNode]:
        """
        Find a path using Theta* algorithm
        
        Args:
            start_node: Starting node
            goal_node: Target node
            
        Returns:
            List[TerrainNode]: Path as a list of nodes
        """
        # Initialize data structures
        open_set = []  # Priority queue
        closed_set = set()
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start_node], id(start_node), start_node))
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current == goal_node:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                path.reverse()
                return path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Check neighbors
            for neighbor, cost in self.terrain_map.get_neighbors(current):
                # Skip if neighbor in closed set
                if neighbor in closed_set:
                    continue
                
                # The key difference from A*: Check line of sight from parent
                if current in came_from and self._has_line_of_sight(came_from[current], neighbor):
                    # If we have line of sight from parent, calculate cost from parent
                    parent = came_from[current]
                    tentative_g = g_score[parent] + parent.position.distance_to(neighbor.position)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        # Path from parent is better
                        came_from[neighbor] = parent
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal_node)
                        
                        if neighbor not in [n for _, _, n in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
                else:
                    # No line of sight or no parent, use standard A* approach
                    tentative_g = g_score[current] + cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal_node)
                        
                        if neighbor not in [n for _, _, n in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
        
        # If we get here, no path was found
        self.logger.warning("Theta* failed to find a path")
        return []
    
    def _has_line_of_sight(self, a: TerrainNode, b: TerrainNode) -> bool:
        """
        Check if there is line of sight between two nodes
        
        Args:
            a: First node
            b: Second node
            
        Returns:
            bool: True if there is line of sight
        """
        # Calculate direction vector
        dx = b.position.x - a.position.x
        dy = b.position.y - a.position.y
        dz = b.position.z - a.position.z
        
        # Calculate distance
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance == 0:
            return True
        
        # Normalize direction
        dx /= distance
        dy /= distance
        dz /= distance
        
        # Number of steps to check (1 step per terrain_map.resolution / 2)
        steps = int(distance / (self.terrain_map.resolution / 2))
        steps = max(steps, 1)  # At least 1 step
        
        # Check points along the line
        for i in range(1, steps):
            t = i / steps
            check_x = a.position.x + dx * distance * t
            check_y = a.position.y + dy * distance * t
            check_z = a.position.z + dz * distance * t
            
            check_pos = Position3D(check_x, check_y, check_z)
            check_node = self.terrain_map.get_nearest_node(check_pos)
            
            if not check_node or not check_node.walkable:
                return False
            
            # Additional check for elevation changes
            if abs(check_node.position.z - (a.position.z + dz * distance * t)) > self.terrain_map.resolution / 2:
                return False
        
        return True
    
    def _heuristic(self, a: TerrainNode, b: TerrainNode) -> float:
        """
        Heuristic function for Theta* pathfinding (3D Euclidean distance)
        
        Args:
            a: First node
            b: Second node
            
        Returns:
            float: Estimated distance between nodes
        """
        return a.position.distance_to(b.position)

class RRT:
    """
    RRT (Rapidly-exploring Random Tree) for pathfinding in complex 3D environments
    Useful for narrow passages and complex terrain where grid-based methods struggle
    """
    
    def __init__(self, terrain_map: TerrainMap, max_iterations: int = 1000, step_size: float = 5.0):
        """
        Initialize RRT
        
        Args:
            terrain_map: TerrainMap to use for pathfinding
            max_iterations: Maximum number of iterations
            step_size: Size of each step
        """
        self.logger = logging.getLogger("wow_ai.decision.advanced_pathfinding.rrt")
        self.terrain_map = terrain_map
        self.max_iterations = max_iterations
        self.step_size = step_size
    
    def find_path(self, start_node: TerrainNode, goal_node: TerrainNode) -> List[TerrainNode]:
        """
        Find a path using RRT algorithm
        
        Args:
            start_node: Starting node
            goal_node: Target node
            
        Returns:
            List[TerrainNode]: Path as a list of nodes
        """
        # Tree structure: node -> parent
        tree = {start_node: None}
        
        # Get bounds for random sampling
        bounds = (
            self.terrain_map.min_bounds,
            self.terrain_map.max_bounds
        )
        
        # RRT loop
        for i in range(self.max_iterations):
            # With some probability, sample the goal directly
            if random.random() < 0.1:
                sample = goal_node.position
            else:
                sample = self._random_sample(bounds)
            
            # Find nearest node in tree
            nearest_node = self._nearest_node(tree.keys(), sample)
            
            # Extend towards sample
            new_node = self._extend(nearest_node, sample)
            
            if not new_node:
                continue
            
            # Add to tree
            tree[new_node] = nearest_node
            
            # Check if we reached the goal
            if new_node.position.distance_to(goal_node.position) < self.step_size:
                # Connect to goal if possible
                if self._has_line_of_sight(new_node, goal_node):
                    tree[goal_node] = new_node
                    
                    # Extract path
                    path = [goal_node]
                    current = goal_node
                    
                    while current in tree and tree[current] is not None:
                        current = tree[current]
                        path.append(current)
                    
                    path.reverse()
                    return path
        
        # If max iterations reached, try to connect closest node to goal
        closest_node = self._nearest_node(tree.keys(), goal_node.position)
        
        if closest_node.position.distance_to(goal_node.position) < self.step_size * 2:
            if self._has_line_of_sight(closest_node, goal_node):
                tree[goal_node] = closest_node
                
                # Extract path
                path = [goal_node]
                current = goal_node
                
                while current in tree and tree[current] is not None:
                    current = tree[current]
                    path.append(current)
                
                path.reverse()
                return path
        
        # No path found
        self.logger.warning("RRT failed to find a path")
        return []
    
    def _random_sample(self, bounds: Tuple[Position3D, Position3D]) -> Position3D:
        """
        Generate a random sample within bounds
        
        Args:
            bounds: (min_bounds, max_bounds) tuple
            
        Returns:
            Position3D: Random position
        """
        min_bounds, max_bounds = bounds
        
        x = random.uniform(min_bounds.x, max_bounds.x)
        y = random.uniform(min_bounds.y, max_bounds.y)
        z = random.uniform(min_bounds.z, max_bounds.z)
        
        return Position3D(x, y, z)
    
    def _nearest_node(self, nodes: Set[TerrainNode], position: Position3D) -> TerrainNode:
        """
        Find nearest node to a position
        
        Args:
            nodes: Set of nodes to search
            position: Target position
            
        Returns:
            TerrainNode: Nearest node
        """
        nearest = None
        min_distance = float('inf')
        
        for node in nodes:
            dist = node.position.distance_to(position)
            if dist < min_distance:
                min_distance = dist
                nearest = node
        
        return nearest
    
    def _extend(self, node: TerrainNode, sample: Position3D) -> Optional[TerrainNode]:
        """
        Extend tree from node towards sample
        
        Args:
            node: Node to extend from
            sample: Sample position
            
        Returns:
            Optional[TerrainNode]: New node or None
        """
        # Calculate direction
        dx = sample.x - node.position.x
        dy = sample.y - node.position.y
        dz = sample.z - node.position.z
        
        # Calculate distance
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance == 0:
            return None
        
        # Normalize and scale by step size
        dx = dx / distance * min(self.step_size, distance)
        dy = dy / distance * min(self.step_size, distance)
        dz = dz / distance * min(self.step_size, distance)
        
        # Calculate new position
        new_pos = Position3D(
            node.position.x + dx,
            node.position.y + dy,
            node.position.z + dz
        )
        
        # Check if valid (inside bounds and terrain exists)
        new_node = self.terrain_map.get_nearest_node(new_pos)
        
        if not new_node or not new_node.walkable:
            return None
        
        # Check line of sight to new node
        if not self._has_line_of_sight(node, new_node):
            return None
        
        return new_node
    
    def _has_line_of_sight(self, a: TerrainNode, b: TerrainNode) -> bool:
        """
        Check if there is line of sight between two nodes
        
        Args:
            a: First node
            b: Second node
            
        Returns:
            bool: True if there is line of sight
        """
        # Calculate direction vector
        dx = b.position.x - a.position.x
        dy = b.position.y - a.position.y
        dz = b.position.z - a.position.z
        
        # Calculate distance
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance == 0:
            return True
        
        # Normalize direction
        dx /= distance
        dy /= distance
        dz /= distance
        
        # Number of steps to check (1 step per terrain_map.resolution / 2)
        steps = int(distance / (self.terrain_map.resolution / 2))
        steps = max(steps, 1)  # At least 1 step
        
        # Check points along the line
        for i in range(1, steps):
            t = i / steps
            check_x = a.position.x + dx * distance * t
            check_y = a.position.y + dy * distance * t
            check_z = a.position.z + dz * distance * t
            
            check_pos = Position3D(check_x, check_y, check_z)
            check_node = self.terrain_map.get_nearest_node(check_pos)
            
            if not check_node or not check_node.walkable:
                return False
        
        return True

class PathfindingManager:
    """
    Manager class that selects the appropriate pathfinding algorithm based on terrain
    """
    
    def __init__(self, terrain_map: TerrainMap):
        """
        Initialize PathfindingManager
        
        Args:
            terrain_map: TerrainMap to use for pathfinding
        """
        self.logger = logging.getLogger("wow_ai.decision.advanced_pathfinding.manager")
        self.terrain_map = terrain_map
        
        # Initialize algorithms
        self.astar = lambda start, goal: self._find_path_astar(start, goal)
        self.jps = JumpPointSearch(terrain_map)
        self.theta_star = ThetaStar(terrain_map)
        self.rrt = RRT(terrain_map)
    
    def find_path(self, start_node: TerrainNode, goal_node: TerrainNode, 
                algorithm: str = "auto") -> List[TerrainNode]:
        """
        Find a path using the selected or auto-selected algorithm
        
        Args:
            start_node: Starting node
            goal_node: Target node
            algorithm: Algorithm to use (auto, astar, jps, theta_star, rrt)
            
        Returns:
            List[TerrainNode]: Path as a list of nodes
        """
        if algorithm == "auto":
            # Select algorithm based on terrain
            algorithm = self._select_algorithm(start_node, goal_node)
        
        # Use the selected algorithm
        if algorithm == "astar":
            return self._find_path_astar(start_node, goal_node)
        elif algorithm == "jps":
            return self.jps.find_path(start_node, goal_node)
        elif algorithm == "theta_star":
            return self.theta_star.find_path(start_node, goal_node)
        elif algorithm == "rrt":
            return self.rrt.find_path(start_node, goal_node)
        else:
            self.logger.warning(f"Unknown algorithm: {algorithm}, using A*")
            return self._find_path_astar(start_node, goal_node)
    
    def _select_algorithm(self, start_node: TerrainNode, goal_node: TerrainNode) -> str:
        """
        Auto-select the best pathfinding algorithm
        
        Args:
            start_node: Starting node
            goal_node: Target node
            
        Returns:
            str: Algorithm name
        """
        # Calculate distance
        distance = start_node.position.distance_to(goal_node.position)
        
        # Check terrain complexity (by sampling points between start and goal)
        dx = goal_node.position.x - start_node.position.x
        dy = goal_node.position.y - start_node.position.y
        dz = goal_node.position.z - start_node.position.z
        
        # Sample 10 points
        terrain_changes = 0
        prev_terrain = None
        
        for i in range(10):
            t = i / 10
            check_x = start_node.position.x + dx * t
            check_y = start_node.position.y + dy * t
            check_z = start_node.position.z + dz * t
            
            check_pos = Position3D(check_x, check_y, check_z)
            check_node = self.terrain_map.get_nearest_node(check_pos)
            
            if check_node:
                if prev_terrain is not None and check_node.terrain_type != prev_terrain:
                    terrain_changes += 1
                prev_terrain = check_node.terrain_type
        
        # Terrain complexity score
        complexity = terrain_changes / 10
        
        # Decision logic
        if distance < 100 and complexity < 0.3:
            # Short distance with simple terrain: A* is efficient enough
            return "astar"
        elif distance < 500 and complexity < 0.5:
            # Medium distance with moderate complexity: JPS for speed
            return "jps"
        elif complexity < 0.7:
            # Any distance with moderate to high complexity: Theta* for smoother paths
            return "theta_star"
        else:
            # Very complex terrain: RRT
            return "rrt"
    
    def _find_path_astar(self, start_node: TerrainNode, goal_node: TerrainNode) -> List[TerrainNode]:
        """
        Find a path using A* algorithm
        
        Args:
            start_node: Starting node
            goal_node: Target node
            
        Returns:
            List[TerrainNode]: Path as a list of nodes
        """
        # Initialize data structures
        open_set = []  # Priority queue
        closed_set = set()
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start_node], id(start_node), start_node))
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current == goal_node:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                path.reverse()
                return path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Check neighbors
            for neighbor, cost in self.terrain_map.get_neighbors(current):
                # Skip if neighbor in closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g score
                tentative_g = g_score[current] + cost
                
                # Check if this path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal_node)
                    
                    # Add to open set if not already there
                    if neighbor not in [n for _, _, n in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
        
        # If we get here, no path was found
        self.logger.warning("A* failed to find a path")
        return []
    
    def _heuristic(self, a: TerrainNode, b: TerrainNode) -> float:
        """
        Heuristic function for pathfinding (3D Euclidean distance)
        
        Args:
            a: First node
            b: Second node
            
        Returns:
            float: Estimated distance between nodes
        """
        return a.position.distance_to(b.position)