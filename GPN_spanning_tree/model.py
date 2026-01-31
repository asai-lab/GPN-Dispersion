import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Part 1: Attention Module
# ============================================================================
class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden)
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        self.Wref = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wq = nn.Linear(n_hidden, n_hidden, bias=False)
    
    def forward(self, q, ref):
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        
        q_proj = self.Wq(q)  # (B, dim)
        ref_proj = self.Wref(ref)
        ref_proj = ref_proj.view(self.batch_size, self.size, self.dim)
        
        q_ex = q_proj.unsqueeze(1).repeat(1, self.size, 1)
        
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        u = torch.bmm(torch.tanh(q_ex + ref_proj), v_view).squeeze(2)
        
        return u, ref_proj

# ============================================================================
# Part 2: LSTM Module
# ============================================================================
class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        self.Wxi = nn.Linear(n_hidden, n_hidden)
        self.Whi = nn.Linear(n_hidden, n_hidden)
        self.wci = nn.Linear(n_hidden, n_hidden)
        self.Wxf = nn.Linear(n_hidden, n_hidden)
        self.Whf = nn.Linear(n_hidden, n_hidden)
        self.wcf = nn.Linear(n_hidden, n_hidden)
        self.Wxc = nn.Linear(n_hidden, n_hidden)
        self.Whc = nn.Linear(n_hidden, n_hidden)
        self.Wxo = nn.Linear(n_hidden, n_hidden)
        self.Who = nn.Linear(n_hidden, n_hidden)
        self.wco = nn.Linear(n_hidden, n_hidden)
    
    def forward(self, x, h, c):
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        h = o * torch.tanh(c)
        return h, c

# ============================================================================
# Part 3: GPN 
# ============================================================================
class GPN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.dim = n_hidden
        self.pointer = Attention(n_hidden)
        self.encoder = LSTM(n_hidden) 
        self.h0 = nn.Parameter(torch.FloatTensor(n_hidden).uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)))
        self.c0 = nn.Parameter(torch.FloatTensor(n_hidden).uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)))
        self.start_token = nn.Parameter(torch.FloatTensor(n_hidden).uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)))
        self.embedding_nodes = nn.Linear(n_feature, n_hidden)
        self.embedding_edges = nn.Linear(n_feature * 2, n_hidden)
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
        self.r1 = nn.Parameter(torch.ones(1))
        self.r2 = nn.Parameter(torch.ones(1))
        self.r3 = nn.Parameter(torch.ones(1))
        self.attention_scale = nn.Parameter(torch.ones(1))  

    def aggregate_neighbors(self, node_embeddings, edge_indices):
        B, N, D = node_embeddings.shape
        device = node_embeddings.device

        # Extract edge endpoint indices
        u_idx = edge_indices[:, 0, :].long()  # [B, E]
        v_idx = edge_indices[:, 1, :].long()  # [B, E]
        
        # Collect neighbor embeddings
        u_emb = torch.gather(node_embeddings, 1, u_idx.unsqueeze(-1).expand(-1, -1, D)) # [B, E, D]
        v_emb = torch.gather(node_embeddings, 1, v_idx.unsqueeze(-1).expand(-1, -1, D)) # [B, E, D]
        
        # Initialize the aggregation result tensor
        neighbor_emb = torch.zeros(B, N, D, device=device)
        edge_counts = torch.zeros(B, N, device=device)

        # Use scatter_add_ for efficient batch aggregation
        # Aggregate v's embeddings to u nodes
        neighbor_emb.scatter_add_(1, u_idx.unsqueeze(-1).expand(-1, -1, D), v_emb)
        # Aggregate the embeddings of u to v nodes
        neighbor_emb.scatter_add_(1, v_idx.unsqueeze(-1).expand(-1, -1, D), u_emb)
        
        # Compute the degree of each node 
        ones = torch.ones_like(u_idx, dtype=torch.float)
        edge_counts.scatter_add_(1, u_idx, ones)
        edge_counts.scatter_add_(1, v_idx, ones)

        # Mean aggregation
        edge_counts = edge_counts.clamp(min=1)
        neighbor_emb = neighbor_emb / edge_counts.unsqueeze(-1)
        
        return neighbor_emb

    def generate_edge_mask(self, nodes_in_tree_mask, all_edge_indices):
        """
        A robust and efficient version that correctly handles mixed batches.
        """
        B, E = all_edge_indices.shape[0], all_edge_indices.shape[2]
        device = nodes_in_tree_mask.device
        assert all_edge_indices.device == device, "Device mismatch between inputs"

        # --- Check and handle initial state ---
        is_first_step = (nodes_in_tree_mask.sum(dim=1) == 0) # shape: [B]

        # If the entire batch is in the initial state, return a mask of all zeros
        if is_first_step.all():
            return torch.zeros(B, E, device=device)

        # --- Calculate the mask for general cases ---
        # Extract edge endpoint indices
        u = all_edge_indices[:, 0, :].long()
        v = all_edge_indices[:, 1, :].long()

        # Collect endpoint states
        is_u_in = nodes_in_tree_mask.gather(1, u)
        is_v_in = nodes_in_tree_mask.gather(1, v)

        # Legal edges: one end in the tree, one end not in
        is_legal_edge = (is_u_in != is_v_in)
        
        # Default mask: assume all edges are illegal
        mask = torch.full((B, E), float('-inf'), device=device)
        # Set the valid positions to 0
        mask[is_legal_edge] = 0.0 

        # --- Force overwrite the mask for initial state samples ---
        # For samples in the batch that are at the first step, their mask should be all 0
        mask[is_first_step] = 0.0
        
        return mask

    def forward(self, last_selected_edge_idx, X_all_nodes, all_edge_features, all_edge_indices, nodes_in_tree_mask, h=None, c=None):
        B, N, _ = X_all_nodes.shape
        E = all_edge_features.shape[1]

        # Initial embeddings
        node_embeddings = self.embedding_nodes(X_all_nodes)  # [B, N, D]
        edge_embeddings = self.embedding_edges(all_edge_features)  # [B, E, D]

        # GNN update
        neighbor_emb = self.aggregate_neighbors(node_embeddings, all_edge_indices)  # [B, N, D]
        context_nodes = node_embeddings.view(B * N, self.dim)
        neighbor_emb_flat = neighbor_emb.view(B * N, self.dim)
        context_nodes = self.r1 * self.W1(context_nodes) + (1-self.r1) * F.relu(self.agg_1(neighbor_emb_flat))
        context_nodes = self.r2 * self.W2(context_nodes) + (1-self.r2) * F.relu(self.agg_2(neighbor_emb_flat))
        context_nodes = self.r3 * self.W3(context_nodes) + (1-self.r3) * F.relu(self.agg_3(neighbor_emb_flat))
        updated_node_embeddings = context_nodes.view(B, N, self.dim)

        # Generate edge embeddings
        u_idx = all_edge_indices[:, 0, :].long()
        v_idx = all_edge_indices[:, 1, :].long()
        b = torch.arange(B, device=updated_node_embeddings.device).unsqueeze(-1)
        u_emb = updated_node_embeddings[b, u_idx, :]
        v_emb = updated_node_embeddings[b, v_idx, :]
        final_edge_embeddings = edge_embeddings + u_emb + v_emb

        # LSTM state initialization
        if h is None or c is None:
            h = self.h0.unsqueeze(0).expand(B, self.dim)
            c = self.c0.unsqueeze(0).expand(B, self.dim)
            mean_node_emb = node_embeddings.mean(dim=1)  # [B, D]
            h, c = self.encoder(mean_node_emb, h, c)

        
        if last_selected_edge_idx is None:
            decoder_input = self.start_token.unsqueeze(0).expand(B, self.dim)
        else:
            batch = torch.arange(B, device=final_edge_embeddings.device)
            decoder_input = final_edge_embeddings[batch, last_selected_edge_idx.long(), :]

        # LSTM update
        h, c = self.encoder(decoder_input, h, c)
        q = h

        # Attention
        u, _ = self.pointer(q, final_edge_embeddings.view(B * E, self.dim))
        edge_mask = self.generate_edge_mask(nodes_in_tree_mask, all_edge_indices)
        u = self.attention_scale * torch.tanh(u) + edge_mask

        return F.softmax(u, dim=1), h, c, final_edge_embeddings