import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import random
from model import GPN

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


def calculate_matching_cost(selected_edge_indices, all_edge_weights):
    selected_weights = all_edge_weights.gather(1, selected_edge_indices)
    total_cost = selected_weights.sum(dim=1)
    return total_cost

# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":

    
    torch.manual_seed(0)
    random.seed(0)

    # --- setting  ---
    MIN_NODES = 2
    MAX_NODES = 40
    n_nodes = list(range(MIN_NODES, MAX_NODES + 1, 2)) 

    FIXED_LR = 5e-4
    FIXED_BATCH_SIZE = 128
    STEPS_PER_EPOCH = 1000
    N_EPOCHS = 100
    N_HIDDEN = 128
    GRAD_CLIP = 2.0

    ALPHA_VALUES = [0 , 1, 3 ,7] 

    
    SAVE_DIR_ROOT = Path("../model/perfect_matching")
    SAVE_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_results = [] 

    for exp_idx, alpha in enumerate(ALPHA_VALUES):
        lr = FIXED_LR
        batch_size = FIXED_BATCH_SIZE

        print(f"\n{'='*60}")
        print(f"{exp_idx+1}/{len(ALPHA_VALUES)}: lr={lr}, alpha={alpha}, batch_size={batch_size}")
        print(f"{'='*60}\n")

        
        current_save_dir = SAVE_DIR_ROOT / f"alpha_{alpha}"
        current_save_dir.mkdir(parents=True, exist_ok=True)

        # --- Initialize model, optimizer, and logger ---
        model = GPN(n_feature=2, n_hidden=N_HIDDEN).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        run_name = f"multi_lr{lr}_alpha{alpha}_bs{batch_size}"
        writer = SummaryWriter(log_dir=current_save_dir / 'runs' / run_name)
        print(f"Hyperparameters: LR={lr}, Alpha={alpha}, BatchSize={batch_size}")

        best_val_cost = float('inf')
        best_epoch = -1

        # --- Training loop ---
        global_step = 0
        for epoch in range(N_EPOCHS):
            model.train()

            epoch_total_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_sampled_cost = 0.0
            epoch_greedy_cost = 0.0

            for i in tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
                optimizer.zero_grad()

                current_n_nodes = random.choice(n_nodes)
                assert current_n_nodes % 2 == 0
                selected_n_edge = current_n_nodes // 2
                nodes = torch.rand(batch_size, current_n_nodes, 2).to(device)
                edge_indices, edge_features, edge_weights_full = prepare_graph_data(nodes, device)

                # ============================================
                #  greedy (Critic/Baseline)
                # ============================================
                model.eval()
                with torch.no_grad():
                    h_greedy, c_greedy = None, None
                    last_selected_edge_idx_greedy = None
                    node_embeddings_cache = None

                    matched_nodes_mask_greedy = torch.zeros(batch_size, current_n_nodes, device=device)
                    selected_indices_greedy_list = []
                    greedy_rewards = []

                    for k in range(selected_n_edge):
                        is_first = (k == 0)
                        current_node_embeddings = node_embeddings_cache

                        probs_greedy, h_greedy, c_greedy, node_emb_greedy = model(
                            nodes, edge_features, edge_indices, h_greedy, c_greedy,
                            matched_nodes_mask_greedy, is_first, last_selected_edge_idx_greedy,
                            node_embeddings=current_node_embeddings
                        )

                        if node_embeddings_cache is None:
                            node_embeddings_cache = node_emb_greedy.detach()

                        selected_edge_idx_greedy = torch.argmax(probs_greedy, dim=1)
                        selected_indices_greedy_list.append(selected_edge_idx_greedy)

                        b_idx = torch.arange(batch_size, device=device)
                        u_greedy = edge_indices[b_idx, 0, selected_edge_idx_greedy].long()
                        v_greedy = edge_indices[b_idx, 1, selected_edge_idx_greedy].long()
                        matched_nodes_mask_greedy.scatter_(1, u_greedy.unsqueeze(1), 1.0)
                        matched_nodes_mask_greedy.scatter_(1, v_greedy.unsqueeze(1), 1.0)
                        last_selected_edge_idx_greedy = selected_edge_idx_greedy
                        edge_weights = edge_weights_full.gather(1, selected_edge_idx_greedy.unsqueeze(1)).squeeze(1)
                        greedy_rewards.append(-edge_weights)

                    R_greedy = torch.stack(greedy_rewards).t().sum(1)

                selected_edges_greedy = torch.stack(selected_indices_greedy_list, dim=1)
                C_cost = calculate_matching_cost(selected_edges_greedy, edge_weights_full)
                    

                # ============================================
                # sampling (Actor)
                # ============================================
                model.train()
                h_sample, c_sample = None, None
                last_selected_edge_idx_sample = None
                matched_nodes_mask_sample = torch.zeros(batch_size, current_n_nodes, device=device)
                log_probs_list = []
                rewards = []
                selected_indices_sample_list = []
                entropies_list = []

                for k in range(selected_n_edge):
                    is_first = (k == 0)
                    current_node_embeddings = node_embeddings_cache 

                    probs_sample, h_sample, c_sample, _ = model(
                        nodes, edge_features, edge_indices, h_sample, c_sample,
                        matched_nodes_mask_sample, is_first, last_selected_edge_idx_sample,
                        node_embeddings=current_node_embeddings 
                    )

                    sampler = torch.distributions.Categorical(probs_sample)
                    selected_edge_idx_sample = sampler.sample()

                    log_probs_list.append(sampler.log_prob(selected_edge_idx_sample))
                    selected_indices_sample_list.append(selected_edge_idx_sample)
                    entropies_list.append(sampler.entropy())
                    edge_weights = edge_weights_full.gather(1, selected_edge_idx_sample.unsqueeze(1)).squeeze(1) # <-- 修正 1
                    rewards.append(-edge_weights)

                    b_idx = torch.arange(batch_size, device=device)
                    u_sample = edge_indices[b_idx, 0, selected_edge_idx_sample].long()
                    v_sample = edge_indices[b_idx, 1, selected_edge_idx_sample].long()
                    matched_nodes_mask_sample.scatter_(1, u_sample.unsqueeze(1), 1.0)
                    matched_nodes_mask_sample.scatter_(1, v_sample.unsqueeze(1), 1.0)
                    last_selected_edge_idx_sample = selected_edge_idx_sample

                selected_edges_sample = torch.stack(selected_indices_sample_list, dim=1)
                R_cost = calculate_matching_cost(selected_edges_sample, edge_weights_full)

                R = torch.stack(rewards).t().sum(1)
                log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1)
                entropies = torch.stack(entropies_list, dim=1).mean(dim=1)

                # ============================================
                # Compute loss and perform update
                # ============================================
                # Self-Critic Baseline: Advantage = Sampled Reward - Greedy Reward
                # Since Reward = -Cost, Advantage = (-R_cost) - (-C_cost) = C_cost - R_cost
                # But loss = -(Advantage * log_probs).mean() 
                # loss = -((C_cost - R_cost) * log_probs).mean() 
                # loss = ((R_cost - C_cost) * log_probs).mean() 
                R_sampled_mean = R.mean().detach()
                R_greedy_mean = R_greedy.mean().detach()
                
                
                baseline = R_greedy.detach() + (R_sampled_mean - R_greedy_mean)

                advantage = R - baseline
                
                policy_loss = -(advantage.detach() * log_probs).mean()
                entropy_loss = entropies.mean()
                loss = policy_loss - alpha * entropy_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                # --- Logging ---
                epoch_total_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                epoch_sampled_cost += R_cost.mean().item()
                epoch_greedy_cost += C_cost.mean().item()

                global_step += 1
                writer.add_scalar('Loss/Total_Step', loss.item(), global_step)
                writer.add_scalar('Loss/Policy_Step', policy_loss.item(), global_step)
                writer.add_scalar('Loss/Entropy_Step', entropy_loss.item(), global_step)
                writer.add_scalar('Reward/Sampled_Cost_Step', R_cost.mean().item(), global_step)
                writer.add_scalar('Reward/Greedy_Cost_Step', C_cost.mean().item(), global_step)


            # --- End of Epoch Logging ---
            avg_epoch_loss = epoch_total_loss / STEPS_PER_EPOCH
            avg_epoch_policy_loss = epoch_policy_loss / STEPS_PER_EPOCH
            avg_epoch_entropy_loss = epoch_entropy_loss / STEPS_PER_EPOCH
            avg_epoch_sampled_cost = epoch_sampled_cost / STEPS_PER_EPOCH
            avg_epoch_greedy_cost = epoch_greedy_cost / STEPS_PER_EPOCH

            writer.add_scalar('Loss/Total_Epoch', avg_epoch_loss, epoch)
            writer.add_scalar('Loss/Policy_Epoch', avg_epoch_policy_loss, epoch)
            writer.add_scalar('Loss/Entropy_Epoch', avg_epoch_entropy_loss, epoch)
            writer.add_scalar('Reward/Sampled_Cost_Epoch', avg_epoch_sampled_cost, epoch)
            writer.add_scalar('Reward/Greedy_Cost_Epoch', avg_epoch_greedy_cost, epoch)

            print(f"\nEpoch {epoch+1} Summary: Avg Loss={avg_epoch_loss:.4f}, Avg Sampled Cost={avg_epoch_sampled_cost:.4f}, Avg Greedy Cost={avg_epoch_greedy_cost:.4f}\n")

            # ============================================
            # 7. Validation
            # ============================================
            model.eval()
            total_val_cost = 0.0
            num_val_batches = 20
            val_samples_count = 0 

            with torch.no_grad():
                for _ in range(num_val_batches):
                    val_n_nodes = random.choice(n_nodes)
                    val_select_n_edge = val_n_nodes // 2
                    nodes_val = torch.rand(batch_size, val_n_nodes, 2).to(device) 
                    edge_indices_val, edge_features_val, edge_weights_val = prepare_graph_data(nodes_val, device)

                    h_val, c_val = None, None
                    last_idx_val = None
                    matched_nodes_mask_val = torch.zeros(batch_size, val_n_nodes, device=device) 
                    selected_indices_val_list = []
                    node_embeddings_cache_val = None

                    for k in range(val_select_n_edge):
                        is_first = (k == 0)
                        current_node_embeddings_val = node_embeddings_cache_val

                        probs_val, h_val, c_val, node_emb_val = model(
                            nodes_val, edge_features_val, edge_indices_val, h_val, c_val,
                            matched_nodes_mask_val, is_first, last_idx_val,
                            node_embeddings=current_node_embeddings_val 
                        )
                        if node_embeddings_cache_val is None:
                           node_embeddings_cache_val = node_emb_val.detach()

                        selected_edge_idx_val = torch.argmax(probs_val, dim=1)
                        selected_indices_val_list.append(selected_edge_idx_val)

                        b_idx = torch.arange(batch_size, device=device) 
                        u_val = edge_indices_val[b_idx, 0, selected_edge_idx_val].long()
                        v_val = edge_indices_val[b_idx, 1, selected_edge_idx_val].long()
                        matched_nodes_mask_val.scatter_(1, u_val.unsqueeze(1), 1.0)
                        matched_nodes_mask_val.scatter_(1, v_val.unsqueeze(1), 1.0)
                        last_idx_val = selected_edge_idx_val

                    selected_edges_val = torch.stack(selected_indices_val_list, dim=1)
                    batch_cost_val = calculate_matching_cost(selected_edges_val, edge_weights_val)
                    total_val_cost += batch_cost_val.sum().item()
                    val_samples_count += batch_size 

            
            avg_val_cost = total_val_cost / val_samples_count if val_samples_count > 0 else float('inf')
            
            writer.add_scalar('Reward/Validation_Cost_Epoch', avg_val_cost, epoch)
            print(f"Epoch {epoch+1} Validation Avg Cost: {avg_val_cost:.4f}")

            # --- Save Best Model ---
            if avg_val_cost < best_val_cost:
                best_val_cost = avg_val_cost
                best_epoch = epoch + 1
                best_model_path = current_save_dir / f"gpn_match_lr{lr}_alpha{alpha}_bs{batch_size}_best.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"*** New best validation cost: {best_val_cost:.4f}. Model saved to {best_model_path} ***")

            # --- Save Checkpoint Every 10 Epochs ---
            if (epoch + 1) % 10 == 0:
                 ckpt_path = current_save_dir / f"gpn_match_lr{lr}_alpha{alpha}_bs{batch_size}_epoch_{epoch+1}.pt"
                 # =======================
                 torch.save({
                     'epoch': epoch + 1,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'best_val_cost': best_val_cost,
                 }, ckpt_path)
                 print(f"Checkpoint saved to {ckpt_path}")

        # === Record experiment results for the current alpha ===
        experiment_results.append({
            'lr': lr,
            'alpha': alpha,
            'batch_size': batch_size,
            'best_val_cost': best_val_cost,
            'best_epoch': best_epoch,
        })
        # =======================================
        writer.close() 

    # === print all result ===
    print("\n" + "="*60)
    print("           TRAINING AND HYPERPARAMETER SEARCH FINISHED")
    print("="*60)
    print("Summary of all experiments:")
    print("-"*60)
    for result in sorted(experiment_results, key=lambda x: x['alpha']):
        print(
            f"alpha={result['alpha']}: "
            f"Best validation cost = {result['best_val_cost']:.4f} at Epoch {result['best_epoch']}"
            f" (lr={result['lr']}, batch_size={result['batch_size']})"
        )
    print("="*60)
