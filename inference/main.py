import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from spanning_tree_solver import MSTSolver
from perfect_matching_solver import MatchingSolver
from others.utils import get_degree_odd_node, calculate_tour_length, convert_tree_to_tsp_tour_recursive

from pathlib import Path


import time


OUTPUT_SOLUTION_DIR = "inference/Results/berlin52"
COST_SCALING_FACTOR = 1e8

def save_as_alg_solution(solutions_list, output_dir, filename, scaling_factor, node_count_N):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    full_path = os.path.join(output_dir, filename)
    
    try:
        with open(full_path, 'w') as f:
            for path, length in solutions_list:
                
                #  float cost -> int cost
                int_cost = int(length * scaling_factor)
                
                path_to_save = []
                
                # Auto-detect if path size is N or N+1
                if len(path) > node_count_N:
                    path_to_save = path[:-1]
                else:
                    path_to_save = path
                
                path_string = "\t".join(map(str, path_to_save))
                
                # write to file
                f.write(f"{int_cost}\t{path_string}\n")
        
        return full_path 
    
    except Exception as e:
        print(f" Failed to save file: {full_path}: {e}")
        return None

def generate_single_sample(coords_tensor, model_low, model_high):

    X = coords_tensor.unsqueeze(0).to(device) 
    batch_size = 1
    problem_size = X.size(1)
    
    tour = []
    mask = torch.zeros(batch_size, problem_size).to(device)
    
    start_idx = 0
    x = X[:, start_idx, :]
    mask[:, start_idx] = -np.inf
    tour.append(start_idx)
    
    h_low, c_low = None, None
    h_high, c_high = None, None

    for _ in range(problem_size - 1):
        with torch.no_grad():
            _, h_low, c_low, latent = model_low(x=x, X_all=X, h=h_low, c=c_low, mask=mask)
            output, h_high, c_high, _ = model_high(x=x, X_all=X, h=h_high, c=c_high, mask=mask, latent=latent.detach())

        sampler = torch.distributions.Categorical(output)
        idx = sampler.sample() 
        
        tour.append(idx.item())
        
        x = X[:, idx.item(), :]
        mask[:, idx.item()] += -np.inf

    return tour


TEST_DATA_PATH = "dataset/berlin52/berlin52_normalized.npy"
GRAPH_INDEX_TO_VIEW = 0

ALPHAS_TO_TEST = [0, 1, 3, 7]
SAMPLING_TREE = 250
SAMPLING_MATCH = 1
SAMPLING_GPN_DIRECT = 250
SAMPLING_RF = 1000

MST_MODEL_DIR = Path("model/spanning_tree")
MST_SUBDIR_TEMPLATE = "alpha_{alpha}" 
MST_FILENAME_TEMPLATE = "gpn_tsp_size40_lr0.0005_alpha{alpha}_batch256_e100.pt"
MATCH_MODEL_BASE_DIR = Path("model/perfect_matching")
MATCH_SUBDIR_TEMPLATE = "alpha_{alpha}" 
MATCH_FILENAME_TEMPLATE = "gpn_match_lr0.0005_alpha{alpha}_bs128_epoch_100.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_graphs_coords = torch.from_numpy(np.load(TEST_DATA_PATH)).to(device, dtype=torch.float32)
nodes_cpu = all_graphs_coords[GRAPH_INDEX_TO_VIEW].to("cpu", dtype=torch.float32)
N = nodes_cpu.shape[0]


start_total_time = time.time()
final_tour = []
for alpha in ALPHAS_TO_TEST:
    mst_model_name = MST_FILENAME_TEMPLATE.format(alpha=alpha)
    mst_model_path = MST_MODEL_DIR / MST_SUBDIR_TEMPLATE.format(alpha=alpha) /mst_model_name
    
    match_model_name = MATCH_FILENAME_TEMPLATE.format(alpha=alpha)
    match_model_path = MATCH_MODEL_BASE_DIR / MATCH_SUBDIR_TEMPLATE.format(alpha=alpha) / match_model_name

    mst_solver = MSTSolver(model_path=mst_model_path, device=device)
    matching_solver = MatchingSolver(model_path=match_model_path, device=device)


    model_tour = []
    dfs_tour = []
    print(f"--- Sampling {SAMPLING_TREE} trees for Alpha={alpha} ({SAMPLING_MATCH} matches per tree)... ---")
    for i in tqdm(range(SAMPLING_TREE), desc=f"Alpha={alpha} ST sampling"):
    
        
        mst_edges = mst_solver.solve(nodes_cpu, use_sampling=True)
        
        odd_nodes_indices = get_degree_odd_node(mst_edges, N)
        odd_node_coords = nodes_cpu[odd_nodes_indices]
        
        for j in range(SAMPLING_MATCH):
            
            #  model match
            global_matching_edges = matching_solver.solve(
                odd_node_coords,
                odd_nodes_indices,
                use_sampling=True
            )
            
            final_edge_set = mst_edges + global_matching_edges
            
            # eulerian_circuit = find_eulerian_cycle(final_edge_set, N)
            
            # final_model_tour = decide_tour(eulerian_circuit, N)
            
            # final_model_cost = calculate_tour_length(final_model_tour, nodes_cpu.numpy())
            final_model_tour, final_model_cost = convert_tree_to_tsp_tour_recursive(final_edge_set, nodes_cpu.numpy())
            
            final_tour.append({'method': f'GPN-TreeM ', 'cost': final_model_cost, 'tour': final_model_tour})

        #dfs
        final_dfs_tour, final_dfs_cost = convert_tree_to_tsp_tour_recursive(mst_edges, nodes_cpu.numpy())
        final_tour.append({'method': f'GPN-Tree ', 'cost': final_dfs_cost, 'tour': final_dfs_tour})




end_inference_time = time.time()

print("\n" + "="*70)
print(f" " * 20 + f"final results")
print("="*70)

results_by_method = {}
for item in final_tour:
    method = item['method']
    if method not in results_by_method:
        results_by_method[method] = {'costs': [], 'tours': []}
    
    results_by_method[method]['costs'].append(item['cost'])
    results_by_method[method]['tours'].append(item['tour'])


print(f"\n--- Analyzing and writing results to {OUTPUT_SOLUTION_DIR}/ ---")
for method in sorted(results_by_method.keys()):
    data = results_by_method[method]
    costs = data['costs']
    tours = data['tours']
    
    solutions_list = []
    for i in range(len(costs)):
        solutions_list.append((tours[i], costs[i])) # (path, length) format
    
    safe_filename = method.replace(" (", "_").replace(")", "").replace(" ", "")
    output_filename = f"{safe_filename}.solution"

    # save solutions to file
    saved_path = save_as_alg_solution(
        solutions_list,
        OUTPUT_SOLUTION_DIR,
        output_filename,
        COST_SCALING_FACTOR,
        N  
    )

print(end_inference_time - start_total_time)