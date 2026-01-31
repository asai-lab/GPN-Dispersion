import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import itertools
import random
from model import GPN

# ---- Prim MST baseline ----
def mst_weight_prim(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [N,2]
    return: MST total weight (scalar)
    """
    N = coords.size(0)
    device = coords.device
    D = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [N,N]
    visited = torch.zeros(N, dtype=torch.bool, device=device)
    visited[0] = True

    min_edge = D[0].clone()
    min_edge[0] = float('inf')

    total = torch.zeros((), device=device)
    for _ in range(N - 1):
        w, j = torch.min(min_edge, dim=0)
        total = total + w
        visited[j] = True
        min_edge = torch.minimum(min_edge, D[j])
        min_edge[visited] = float('inf')
    return total

def prepare_graph_data(nodes, device="cuda"):
    """
    Generate all edge indices, edge features, and edge weights for a batch of N nodes in a graph.
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
# main
# ============================================================================
if __name__ == "__main__":
    # --- 1. setting  ---
    parser = argparse.ArgumentParser(description="GPN for Spanning Tree with Hyperparameter Search")
    parser.add_argument('--size',  default=40, type=int, help="size of TSP")
    parser.add_argument('--search_mode', default='grid', choices=['grid', 'random'], help="Search mode: grid or random")
    parser.add_argument('--n_experiments', default=10, type=int, help="Number of experiments for random search")
    parser.add_argument('--epoch', default=50, type=int, help="Number of epochs")
    parser.add_argument('--steps_per_epoch', default=1000, type=int, help='Steps per epoch')
    args = vars(parser.parse_args())

    size = args['size']

   
    param_grid = {
        'lr': [5e-4],
        'alpha': [0 , 1, 3, 7],
        'batch_size': [256]
    }

    search_mode = args['search_mode']
    if search_mode == 'grid':
        param_combinations = list(itertools.product(
            param_grid['lr'], param_grid['alpha'], param_grid['batch_size']
        ))
    else:
        param_combinations = [
            (
                random.choice(param_grid['lr']),
                random.choice(param_grid['alpha']),
                random.choice(param_grid['batch_size'])
            )
            for _ in range(args['n_experiments'])
        ]

    save_dir = Path('../model/spanning_tree')
    save_dir.mkdir(parents=True, exist_ok=True)

    # save experiment results
    experiment_results = []


    for exp_idx, (lr, alpha, batch_size) in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"Experiment {exp_idx+1}/{len(param_combinations)}: lr={lr}, alpha={alpha}, batch_size={batch_size}")
        print(f"{'='*60}\n")

        # TensorBoard
        run_name = f"lr{lr}_alpha{alpha}_batch{batch_size}"
        writer = SummaryWriter(log_dir=save_dir / 'runs' / run_name)

        best_val_tour_len = float('inf')
        best_epoch = -1
        B = batch_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- Create model and optimizer ---
        model = GPN(n_feature=2, n_hidden=128).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # --- training ---
        for epoch in range(args['epoch']):
            model.train()
            
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_total_loss = 0.0
            epoch_rewards = []

            for i in tqdm(range(args['steps_per_epoch']), desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
            
                # --- prepare data ---
                X_nodes = torch.rand(B, size, 2).to(device)
                edge_indices_full, edge_features_full, edge_weights_full = prepare_graph_data(X_nodes)


                # --- Greedy baseline ---
                model.eval()
                with torch.no_grad():

                    greedy_nodes_in_tree = torch.zeros(B, size, dtype=torch.bool, device=device)
                    greedy_start_node_idx = torch.randint(0, size, (B,), device=device)
                    greedy_nodes_in_tree.scatter_(1, greedy_start_node_idx.unsqueeze(1), True)

                    g_h, g_c = None, None
                    greedy_last_selected_edge_idx = None
                    greedy_rewards = []

                    for k in range(size - 1):
                        g_output, g_h, g_c, greedy_final_edge_embeddings = model(
                            greedy_last_selected_edge_idx, X_nodes, edge_features_full, 
                            edge_indices_full, greedy_nodes_in_tree, g_h, g_c
                        )
                        
                        greedy_selected_edge_idx = torch.argmax(g_output, dim=1)


                        # Calculate edge weights and rewards
                        edge_weights = edge_weights_full.gather(1, greedy_selected_edge_idx.unsqueeze(1)).squeeze(1)
                        greedy_rewards.append(-edge_weights)

                        endpoints_idx = edge_indices_full[torch.arange(B, device=device), :, greedy_selected_edge_idx]

                        u_selected = endpoints_idx[:, 0]
                        v_selected = endpoints_idx[:, 1]
                        is_u_in = greedy_nodes_in_tree.gather(1, u_selected.unsqueeze(1).long()).squeeze(1)
                        new_node_idx = torch.where(is_u_in, v_selected, u_selected)
                        greedy_nodes_in_tree.scatter_(1, new_node_idx.unsqueeze(1).long(), True)
                        
                        greedy_last_selected_edge_idx = greedy_selected_edge_idx
                    
                    R_greedy = torch.stack(greedy_rewards).t().sum(1)



                # --- Sampled baseline ---
                model.train()

                nodes_in_tree = torch.zeros(B, size, dtype=torch.bool, device=device)
                start_node_idx = torch.randint(0, size, (B,), device=device)
                nodes_in_tree.scatter_(1, start_node_idx.unsqueeze(1), True)
                

                h, c = None, None
                last_selected_edge_idx = None

                logprobs = []
                rewards = []
                entropies = []

                for k in range(size - 1):
                    output, h, c, final_edge_embeddings = model(
                        last_selected_edge_idx, X_nodes, edge_features_full, 
                        edge_indices_full, nodes_in_tree, h, c
                    )
                    
                    sampler = torch.distributions.Categorical(output)
                    selected_edge_idx = sampler.sample()

                    logprobs.append(sampler.log_prob(selected_edge_idx))
                    entropies.append(sampler.entropy())


                    edge_weights = edge_weights_full.gather(1, selected_edge_idx.unsqueeze(1)).squeeze(1) 
                    rewards.append(-edge_weights)

                    endpoints_idx = edge_indices_full[torch.arange(B, device=device), :, selected_edge_idx]

                    u_selected = endpoints_idx[:, 0]
                    v_selected = endpoints_idx[:, 1]
                    
                    # Check if the u node is already in the tree, resulting in a [B] shaped boolean tensor
                    is_u_in = nodes_in_tree.gather(1, u_selected.unsqueeze(1).long()).squeeze(1)
                    
                    # Use torch.where to find the new node, resulting in a [B] shaped index tensor
                    new_node_idx = torch.where(is_u_in, v_selected, u_selected)
                    
                    # Update nodes_in_tree mask, passing in [B, 1] index
                    nodes_in_tree.scatter_(1, new_node_idx.unsqueeze(1).long(), True)
                    
                    last_selected_edge_idx = selected_edge_idx


                R = torch.stack(rewards).t().sum(1)
                logprobs = torch.stack(logprobs).t().sum(1)
                entropies = torch.stack(entropies).t().mean(1)

                # --- Central Self-Critic Baseline ---
                #  baseline = R_greedy + (R_sampled_mean - R_greedy_mean)

                R_sampled_mean = R.mean().detach()
                R_greedy_mean = R_greedy.mean().detach()
                
                # 原 baseline = R.mean().detach() 
                baseline = R_greedy.detach() + (R_sampled_mean - R_greedy_mean)

                advantage = R - baseline
                
                policy_loss = -(advantage.detach() * logprobs).mean()
                entropy_loss = entropies.mean()
                loss = policy_loss - alpha * entropy_loss
                
                epoch_policy_loss += policy_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                epoch_total_loss += loss.item()
                epoch_rewards.append(R)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            # --- Calculate average loss and rewards ---
            epoch_policy_loss /= args['steps_per_epoch']
            epoch_entropy_loss /= args['steps_per_epoch']
            epoch_total_loss /= args['steps_per_epoch']
            epoch_rewards = torch.cat(epoch_rewards).mean().item()

            # --- Evaluation ---
            model.eval()
            with torch.no_grad():
                X_val = torch.rand(B, size, 2).to(device)
                edge_indices_val, edge_features_val, edge_weights_val = prepare_graph_data(X_val)

                nodes_in_tree_val = torch.zeros(B, size, dtype=torch.bool, device=device)
                start_node_idx_val = torch.randint(0, size, (B,), device=device)
                nodes_in_tree_val.scatter_(1, start_node_idx_val.unsqueeze(1), True)

                h_val, c_val = None, None
                last_idx_val = None
                val_rewards = []
                
                for _ in range(size - 1):
                    output, h_val, c_val, _ = model(
                        last_idx_val, X_val, edge_features_val, edge_indices_val,
                        nodes_in_tree_val, h_val, c_val
                    )
                    selected_edge_idx = torch.argmax(output, dim=1)
                    

                    edge_weights = edge_weights_val.gather(1, selected_edge_idx.unsqueeze(1)).squeeze(1) 
                    val_rewards.append(-edge_weights)
                    
                    endpoints_idx = edge_indices_val[torch.arange(B, device=device), :, selected_edge_idx]
                    u_selected_val = endpoints_idx[:, 0]
                    v_selected_val = endpoints_idx[:, 1]
                    is_u_in_val = nodes_in_tree_val.gather(1, u_selected_val.unsqueeze(1).long()).squeeze(1)
                    new_node_idx_val = torch.where(is_u_in_val, v_selected_val, u_selected_val)
                    nodes_in_tree_val.scatter_(1, new_node_idx_val.unsqueeze(1).long(), True)

                val_tree_weight = -torch.stack(val_rewards).t().sum(1).mean()

                
                if val_tree_weight < best_val_tour_len:
                    best_val_tour_len = val_tree_weight
                    best_epoch = epoch
                    best_ckpt_path = save_dir / f"gpn_tsp_size{size}_lr{lr}_alpha{alpha}_batch{batch_size}_best.pt"
                    print(f"Saving best model to: {best_ckpt_path}")
                    torch.save(model.state_dict(), best_ckpt_path)

               
                if (epoch + 1) % 10 == 0:
                    ckpt_path = save_dir / f"gpn_tsp_size{size}_lr{lr}_alpha{alpha}_batch{batch_size}_e{epoch+1}.pt"
                    print(f"Saving checkpoint to: {ckpt_path}")
                    torch.save(model.state_dict(), ckpt_path)
                
                # --- MST baseline ---
                mst_mean = torch.stack([mst_weight_prim(X_val[b]) for b in range(B)]).mean()

                # --- TensorBoard ---
                writer.add_scalar('Loss/Policy', epoch_policy_loss, epoch)
                writer.add_scalar('Loss/Entropy', epoch_entropy_loss, epoch)
                writer.add_scalar('Loss/Total', epoch_total_loss, epoch)
                writer.add_scalar('Reward/Train', -epoch_rewards, epoch)
                writer.add_scalar('Reward/Val', val_tree_weight.item(), epoch)
                writer.add_scalar('Reward/MST_Baseline', mst_mean.item(), epoch)
                writer.add_scalar('Gradient/Norm', grad_norm, epoch)

                print(
                    f"Epoch {epoch+1}: "
                    f"Avg Sampled Tree Weight = {-epoch_rewards:.4f} | "
                    f"Avg Greedy Val Weight = {val_tree_weight.item():.4f} | "
                    f"MST Baseline = {mst_mean.item():.4f} | "
                    f"Policy Loss = {epoch_policy_loss:.4f} | "
                    f"Entropy Loss = {epoch_entropy_loss:.4f} | "
                    f"Total Loss = {epoch_total_loss:.4f} | "
                    f"Gradient Norm = {grad_norm:.4f}"
                )

        # result record
        experiment_results.append({
            'size': size,
            'lr': lr,
            'alpha': alpha,
            'batch_size': batch_size,
            'best_val_tour_len': best_val_tour_len,
            'best_epoch': best_epoch,
            'mst_baseline': mst_mean.item()
        })

        
        writer.close()

    # --- print all result ---
    print("\n" + "="*60)
    print("           TRAINING AND HYPERPARAMETER SEARCH FINISHED")
    print("="*60)
    print("Summary of all experiments:")
    print("-"*60)
    for result in experiment_results:
        print(
            f"size={result['size']}, lr={result['lr']}, alpha={result['alpha']}, "
            f"batch_size={result['batch_size']}: "
            f"Best Val Weight = {result['best_val_tour_len']:.4f} at Epoch {result['best_epoch']}, "
            f"MST Baseline = {result['mst_baseline']:.4f}"
        )
    print("="*60)