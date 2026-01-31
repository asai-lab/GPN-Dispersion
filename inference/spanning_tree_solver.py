import torch
import numpy as np
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GPN_spanning_tree.model import GPN
from others.utils import prepare_graph_data

class MSTSolver:
    
    def __init__(self, model_path, n_hidden=128, device=None):
        """
        Load model
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"spanning tree: Using device: {self.device}")
        print(f"spanning tree: Loading model state_dict: {model_path}")

        self.model = GPN(n_feature=2, n_hidden=n_hidden).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            raise RuntimeError(f"Error occurred while loading the spanning tree model state_dict: {e}\n"
                               f"Please check if the file '{model_path}' exists.")

        self.model.eval() 
        print("MST Solver: Model loading completed")
        self.n_hidden = n_hidden

    @torch.no_grad()
    def solve(self, nodes_cpu, use_sampling=True, start_node=0):
        """
        Generates a spanning tree for a single graph.

        Args:
            nodes_cpu (torch.Tensor or np.ndarray): Node coordinates of the single graph, shape [N, 2].
            use_sampling (bool): Whether to use sampling for generation (True=sampling, False=greedy).
            start_node (int): The starting node index for the spanning tree.

        Returns:
            list: A list of edges forming the generated spanning tree.  
        """
        if isinstance(nodes_cpu, np.ndarray):
            nodes_cpu = torch.from_numpy(nodes_cpu).float()

        N = nodes_cpu.shape[0]
        if N <= 1:
             return []

        num_edges_to_select = N - 1

        # --- perpare data ---
        X_nodes = nodes_cpu.unsqueeze(0).to(self.device)
        edge_indices, edge_features, _ = prepare_graph_data(X_nodes, self.device)

        # --- Initialize state ---
        h, c = None, None
        last_idx = None
        nodes_in_tree_mask = torch.zeros(1, N, device=self.device)
        valid_start_node = max(0, min(start_node, N-1))
        nodes_in_tree_mask[0, valid_start_node] = 1.0

        selected_edge_indices_list = []
        mst_edges = []

        
        for _ in range(num_edges_to_select):
            
            probs, h, c, _ = self.model(
                last_idx, X_nodes, edge_features, edge_indices, 
                nodes_in_tree_mask.float(), 
                h, c
            )

            # --- select edge ---
            if use_sampling:
                sampler = torch.distributions.Categorical(probs)
                selected_edge_idx = sampler.sample()[0]
            else:
                selected_edge_idx = torch.argmax(probs, dim=1)[0]

            selected_edge_indices_list.append(selected_edge_idx.item())

            # --- update mask ---
            endpoint_pair = edge_indices[0, :, selected_edge_idx.item()]
            u, v = endpoint_pair[0].item(), endpoint_pair[1].item()
            mst_edges.append([u, v])

            is_u_in = nodes_in_tree_mask[0, u] 
            new_node = v if is_u_in else u
            nodes_in_tree_mask[0, new_node] = True 

            last_idx = selected_edge_idx

        return mst_edges