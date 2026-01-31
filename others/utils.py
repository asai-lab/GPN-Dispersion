import torch
import numpy as np
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random
from collections import defaultdict
from scipy.spatial import distance
import itertools
import matplotlib.pyplot as plt 
# ============================================================================
# Get graph information
# ============================================================================
def prepare_graph_data(nodes, device="cuda"):
    """
    Generate all edge indices, edge features, and edge weights for a batch of graphs with N nodes.
    """

    B, N, _ = nodes.shape
    idx_pairs = torch.combinations(torch.arange(N, device=device), r=2) # [E, 2]
    E = idx_pairs.shape[0]
    all_edge_indices = idx_pairs.unsqueeze(0).expand(B, -1, -1) # [B, E, 2]
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, E)
    u_idx = all_edge_indices[..., 0]
    v_idx = all_edge_indices[..., 1]
    u_coords = nodes[b_idx, u_idx]
    v_coords = nodes[b_idx, v_idx]
    edge_features = torch.cat([u_coords, v_coords], dim=-1) # [B, E, 4]
    edge_weights = torch.norm(u_coords - v_coords, dim=-1) # [B, E]
    all_edge_indices_transposed = all_edge_indices.transpose(1, 2).contiguous() # [B, 2, E]

    return all_edge_indices_transposed, edge_features, edge_weights


# ============================================================================
# Extract nodes with odd degrees
# ============================================================================
def get_degree_odd_node(mst_edge, N):
    all_nodes_in_edges = torch.tensor(list(mst_edge), dtype=torch.long).flatten()

    degrees = torch.bincount(all_nodes_in_edges, minlength=N)

    odd_mask = (degrees % 2) != 0

    odd_nodes_indices = torch.where(odd_mask)[0]
    
    return odd_nodes_indices


# ============================================================================
# Find Eulerian cycle
# ============================================================================
def find_eulerian_cycle(edges_list, N):
    """
    Args:
        edges_list (list): List containing all edges
        N (int): Total number of nodes
        
    Returns:
        list: A cycle containing the node visit order, e.g., [0, 1, 3, 2, 1, 0]
    """
    
    # Create a "MultiGraph"
    # This is "MUST"! Because edges from ST and Matching might overlap
    # If nx.Graph() is used, overlapping edges will be lost
    G = nx.MultiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges_list)

    # Check if the graph is truly Eulerian
    if not nx.is_eulerian(G):
        raise ValueError("The combined graph is not Eulerian (may have odd degree nodes or be disconnected)")
        
    # Find Eulerian cycle (starting from node 0)
    cycle_edges = list(nx.eulerian_circuit(G, source=0))
    
    # Convert format
    # (0, 1), (1, 2), (2, 0) -> [0, 1, 2, 0]
    
    if not cycle_edges:
        return [0] # Handle edge case for N=1 with only one node

    cycle_nodes = [u for u, v in cycle_edges]
    cycle_nodes.append(cycle_edges[-1][1])
    
    return cycle_nodes

# ============================================================================
# Pruning (Eulerian cycle -> TSP tour)
# ============================================================================
def decide_tour(cycle, N):
    """
    Args:
        cycle (list): Eulerian cycle
        N (int): Total number of nodes
    Returns:
        list: A TSP path
    """
    visited = [False] * N
    tour = []
    for node in cycle:
        if not visited[node]:
            tour.append(node)
            visited[node] = True
            
    tour.append(tour[0]) 

    return tour

# ============================================================================
# Calculate TSP tour length
# ============================================================================

def calculate_tour_length(tour_nodes, nodes_coords):
    """
    Calculate the total length of a closed TSP tour.
    
    Args:
        tour_nodes (list): TSP tour (e.g., [0, 1, 3, 2, 0])
        nodes_coords (np.ndarray): Node coordinates (Shape [N, 2])
        
    Returns:
        float: Total length of the tour
    """
    tour_length = 0.0
    
    for i in range(len(tour_nodes) - 1):
        u = tour_nodes[i]
        v = tour_nodes[i+1]
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(nodes_coords[u] - nodes_coords[v])
        tour_length += distance
        
    return tour_length

# ============================================================================
# Spanning Tree -> TSP tour
# ============================================================================
def convert_tree_to_tsp_tour_recursive(tree_edges, coords):
    """
    Convert a spanning tree to a TSP Tour.
    Uses "recursive" DFS, starting fixed at node 0, but re-samples randomly from "remaining" options at each branch.

    Args:
        tree_edges (list of lists): List of tree edges.
        coords (np.array): Coordinates of all nodes.

    Returns:
        tuple: (list, float)
               - path (list): Node visit order of the TSP tour.
               - tour_cost (float): Total length (cost) of the tour.
    """
    if not tree_edges or len(coords) == 0:
        return [], 0.0
    
    problem_size = len(coords)
    
    # Create adjacency list
    adj = defaultdict(list)
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Prepare variables for recursion
    path = []
    visited = [False] * problem_size

    # Define recursive core function
    def _dfs_recursive(node):
        # Mark current node as visited and add to final path
        visited[node] = True
        path.append(node)
        
        # This while loop is key to implementing "re-sampling after return"
        while True:
            # 1. Re-check "every time" which neighbors from the current node haven't been visited
            unvisited_neighbors = [n for n in adj[node] if not visited[n]]

            # 2. If no unvisited neighbors, this branch is fully explored, break loop
            if not unvisited_neighbors:
                break
            
            # 3. Randomly (Uniform) pick one from the "remaining" options
            chosen_neighbor = random.choice(unvisited_neighbors)
            
            # 4. Recursively explore the chosen neighbor
            _dfs_recursive(chosen_neighbor)
            
    # Start recursion from fixed starting point
    # first = random.randint(0, problem_size - 1) 
    _dfs_recursive(0)
    
    if len(path) != problem_size:
        unvisited = [i for i, v in enumerate(visited) if not v]
        path.extend(unvisited)
        
    tour_cost = 0.0
    for i in range(problem_size):
        start_node_idx = path[i]
        end_node_idx = path[(i + 1) % problem_size] 
        tour_cost += np.linalg.norm(coords[start_node_idx] - coords[end_node_idx])
        
    return path, tour_cost


# ============================================================================
# Jaccard index calculation
# ============================================================================


def path_to_edge_set(path_nodes):
    """Convert TSP path (node list) to standardized edge set"""
    edges = set()
    # Assume path is closed, e.g., [0, 1, 3, 2, 0]
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        edges.add(tuple(sorted((u, v))))
    return edges

def calculate_average_jaccard_for_paths(list_of_paths):
    """
    Calculate the average pairwise Jaccard Index for a list of TSP paths (list of lists).
    
    Args:
        list_of_paths (list): A list where each element is a TSP path (node list).
                              e.g., [[0,1,2,0], [0,2,1,0]]
                              
    Returns:
        (float, list): Average Jaccard Index and list of all pairwise scores
    """
    
    path_edge_sets = [path_to_edge_set(path) for path in list_of_paths]

    if len(path_edge_sets) < 2:
        print("Warning: Less than 2 paths, cannot calculate Jaccard Index.")
        return 0.0, []

    jaccard_scores = []
    
    for set_A, set_B in itertools.combinations(path_edge_sets, 2):
        intersection = set_A.intersection(set_B)
        union = set_A.union(set_B)
        
        if not union:
            score = 1.0 # If both are empty sets
        else:
            score = len(intersection) / len(union)
            
        jaccard_scores.append(score)

    if not jaccard_scores:
        return 0.0, []
        
    average_score = sum(jaccard_scores) / len(jaccard_scores)
    return average_score, jaccard_scores



