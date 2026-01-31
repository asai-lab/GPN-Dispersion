import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ============================================================================
# 1. Attention
# ============================================================================
class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.dim = n_hidden
        
        v = torch.FloatTensor(n_hidden)
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    def forward(self, q, ref): 
        batch_size = q.size(0)
        size = ref.size(1) 
        
        q = self.Wq(q) 
        ref = self.Wref(ref) 
        
        q_ex = q.unsqueeze(1).repeat(1, size, 1)
        
        v_view = self.v.unsqueeze(0).expand(batch_size, self.dim).unsqueeze(2)
        
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2) 
        
        return u, ref 

# ============================================================================
# 2. LSTM
# ============================================================================
class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)
        self.Whi = nn.Linear(n_hidden, n_hidden)
        self.wci = nn.Linear(n_hidden, n_hidden)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)
        self.Whf = nn.Linear(n_hidden, n_hidden)
        self.wcf = nn.Linear(n_hidden, n_hidden)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)
        self.Whc = nn.Linear(n_hidden, n_hidden)
        
        # parameters for output gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)
        self.Who = nn.Linear(n_hidden, n_hidden)
        self.wco = nn.Linear(n_hidden, n_hidden)
    
    def forward(self, x, h, c): # query and reference
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        h = o * torch.tanh(c)
        return h, c


# ============================================================================
# 3. GPN model
# ============================================================================
class GPN(nn.Module):  
    
    def __init__(self, n_feature, n_hidden):
        
        super(GPN, self).__init__()
        self.dim = n_hidden
        
        self.pointer = Attention(n_hidden) 
        self.encoder = LSTM(n_hidden)      
        
        self.h0 = nn.Parameter(torch.FloatTensor(n_hidden).uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)))
        self.c0 = nn.Parameter(torch.FloatTensor(n_hidden).uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)))
        
        self.start_token = nn.Parameter(torch.FloatTensor(n_hidden).uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)))

        self.embedding_nodes = nn.Linear(n_feature, n_hidden)
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
        self.r1 = nn.Parameter(torch.ones(1))
        self.r2 = nn.Parameter(torch.ones(1))
        self.r3 = nn.Parameter(torch.ones(1))
        
        self.embedding_edges = nn.Linear(n_feature * 2, n_hidden)
        self.attention_scale = nn.Parameter(torch.ones(1)) 
        

    def aggregate_neighbors(self, node_embeddings, edge_indices):
        B, N, D = node_embeddings.shape
        device = node_embeddings.device

        u_idx = edge_indices[:, 0, :].long()  
        v_idx = edge_indices[:, 1, :].long()  
        
        u_emb = torch.gather(node_embeddings, 1, u_idx.unsqueeze(-1).expand(-1, -1, D)) 
        v_emb = torch.gather(node_embeddings, 1, v_idx.unsqueeze(-1).expand(-1, -1, D)) 
        
        neighbor_emb = torch.zeros(B, N, D, device=device)
        edge_counts = torch.zeros(B, N, device=device)

        neighbor_emb.scatter_add_(1, u_idx.unsqueeze(-1).expand(-1, -1, D), v_emb)
        neighbor_emb.scatter_add_(1, v_idx.unsqueeze(-1).expand(-1, -1, D), u_emb)
        
        ones = torch.ones_like(u_idx, dtype=torch.float)
        edge_counts.scatter_add_(1, u_idx, ones)
        edge_counts.scatter_add_(1, v_idx, ones)

        edge_counts = edge_counts.clamp(min=1)
        neighbor_emb = neighbor_emb / edge_counts.unsqueeze(-1)
        
        return neighbor_emb


    def generate_edge_mask(self, matched_nodes_mask, all_edge_indices):
        """
        Only allow edges where both endpoints are unmatched.
        """
        B, E = all_edge_indices.shape[0], all_edge_indices.shape[2]
        device = matched_nodes_mask.device

        # Check if it's the first step
        is_first_step = (matched_nodes_mask.sum(dim=1) == 0)

        if is_first_step.all():
            return torch.zeros(B, E, device=device)

        u = all_edge_indices[:, 0, :].long()
        v = all_edge_indices[:, 1, :].long()

        is_u_matched = matched_nodes_mask.gather(1, u) > 0  
        is_v_matched = matched_nodes_mask.gather(1, v) > 0

        is_legal_edge = (~is_u_matched) & (~is_v_matched)
        
        mask = torch.full((B, E), float('-inf'), device=device)
        mask[is_legal_edge] = 0.0 

        mask[is_first_step] = 0.0
        
        return mask

    def forward(self, node_coords, edge_features, edge_indices, h, c, matched_nodes_mask, 
                is_first_step=False, last_selected_edge_idx=None, node_embeddings=None): 
        
        B, N, _ = node_coords.shape
        E = edge_features.shape[1]
        
        # GNN computation: only calculate when node_embeddings is not provided
        if node_embeddings is None:
            embedded_nodes = self.embedding_nodes(node_coords)  # (B, N, dim)
            
            neighbor_emb = self.aggregate_neighbors(embedded_nodes, edge_indices)
            
            context = embedded_nodes.view(B * N, self.dim)
            neighbor_flat = neighbor_emb.view(B * N, self.dim)
            
            context = self.r1 * self.W1(context) + (1 - self.r1) * F.relu(self.agg_1(neighbor_flat))
            context = self.r2 * self.W2(context) + (1 - self.r2) * F.relu(self.agg_2(neighbor_flat))
            context = self.r3 * self.W3(context) + (1 - self.r3) * F.relu(self.agg_3(neighbor_flat))
            
            node_embeddings = context.view(B, N, self.dim)
        
        # Edge embeddings
        edge_embeddings = self.embedding_edges(edge_features)  # (B, E, dim)
        
        # Fuse endpoint embeddings into edges
        u_idx = edge_indices[:, 0, :].long()
        v_idx = edge_indices[:, 1, :].long()
        b = torch.arange(B, device=node_coords.device).unsqueeze(1).expand(-1, E)
        u_emb = node_embeddings[b, u_idx]
        v_emb = node_embeddings[b, v_idx]
        final_edge_embeddings = edge_embeddings + u_emb + v_emb
        
        # LSTM state initialization
        if is_first_step:
            h = self.h0.unsqueeze(0).repeat(B, 1)
            c = self.c0.unsqueeze(0).repeat(B, 1)
            # Initialize context with mean node embeddings
            mean_node_emb = node_embeddings.mean(dim=1)
            h, c = self.encoder(mean_node_emb, h, c)
            decoder_input = self.start_token.unsqueeze(0).repeat(B, 1)
        else:
            decoder_input = final_edge_embeddings.gather(
                1, 
                last_selected_edge_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.dim)
            ).squeeze(1)

        h, c = self.encoder(decoder_input, h, c) 

        # Attention
        logits, _ = self.pointer(h, final_edge_embeddings) 

        # Apply mask and scale
        edge_mask = self.generate_edge_mask(matched_nodes_mask, edge_indices)
        logits = self.attention_scale * torch.tanh(logits) + edge_mask

        probabilities = F.softmax(logits, dim=1)

        return probabilities, h, c, node_embeddings