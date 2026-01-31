import torch
import numpy as np
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GPN_perfect_matching.model import GPN 
from others.utils import prepare_graph_data

class MatchingSolver:
    
    def __init__(self, model_path, n_hidden=128, device=None):
        """
        Load model
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Matching Solver: Using device: {self.device}")
        print(f"Matching Solver: Loading model state_dict: {model_path}")

        self.model = GPN(n_feature=2, n_hidden=n_hidden).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Error occurred while loading the Matching model state_dict: {e}\n"
                               f"Please check if the file '{model_path}' exists.")

        self.model.eval() 
        print("Matching Solver: Model loading completed.")
        self.n_hidden = n_hidden

    @torch.no_grad()
    def solve(self, nodes_cpu, global_indices_tensor, use_sampling=True):
        """
        Generate

        Args:
            nodes_cpu (torch.Tensor or np.ndarray): Coordinates of the nodes, shape [K, 2]. K must be even.
            use_sampling (bool): Whether to use sampling for generation (True=sampling, False=greedy).

        Returns:
            match list
        """
        
        if isinstance(nodes_cpu, np.ndarray):
            nodes_cpu = torch.from_numpy(nodes_cpu).float()

        K = nodes_cpu.shape[0]
        
        if K <= 0:
             return []
        
        # check
        if K % 2 != 0:
            raise ValueError(f"Matching problem must have an even number of nodes, but received {K}.")
            
        num_edges_to_select = K // 2
        

        # --- perpare data ---
        X_nodes = nodes_cpu.unsqueeze(0).to(self.device)
        edge_indices, edge_features, _ = prepare_graph_data(X_nodes, self.device)

        # --- Initialize state ---
        h, c = None, None
        last_idx = None
        
        matched_nodes_mask = torch.zeros(1, K, device=self.device, dtype=torch.float)
        
        node_embeddings_cache = None
        selected_edge_indices_list = []
        matching_edges_local = [] 

        for k in range(num_edges_to_select):
            is_first = (k == 0)
            current_node_embeddings = node_embeddings_cache

            
            probs, h, c, node_emb = self.model(
                X_nodes, edge_features, edge_indices, h, c,
                matched_nodes_mask,
                is_first, last_idx,
                node_embeddings=current_node_embeddings
            )

            if node_embeddings_cache is None:
                node_embeddings_cache = node_emb.detach()

            # --- select edge ---
            if use_sampling:
                sampler = torch.distributions.Categorical(probs)
                selected_edge_idx = sampler.sample()
            else:
                selected_edge_idx = torch.argmax(probs, dim=1)

            selected_edge_indices_list.append(selected_edge_idx.item())

            # --- update mask ---
            endpoint_pair = edge_indices[0, :, selected_edge_idx.item()]
            u, v = endpoint_pair[0].item(), endpoint_pair[1].item()
            
            matching_edges_local.append([u, v])

            # Mark u and v as matched
            matched_nodes_mask[0, u] = 1.0
            matched_nodes_mask[0, v] = 1.0

            last_idx = selected_edge_idx

        # --- map to global indices ---
        global_edges = []
        mapping = global_indices_tensor.cpu() 
    
        for u_local, v_local in matching_edges_local:
            
            u_global = mapping[u_local].item() 
            v_global = mapping[v_local].item() 
        
            global_edges.append([u_global, v_global])

        return global_edges