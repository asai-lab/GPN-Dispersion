import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from collections import Counter
import sys
import argparse
import itertools
from others.utils import *
import os
import numpy as np 


def standardize_edge(u, v):
    return tuple(sorted((u, v)))

def load_raw_solutions(filepath):
    print(f"--- Step 1: Loading original candidate pool from '{filepath}' ---")
    raw_pool = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line: continue
                parts = line.split()
                try:
                    cost = float(parts[0]) 
                    node_path = parts[1:] 
                    edge_set = set()
                    for i in range(len(node_path) - 1):
                        edge_set.add(standardize_edge(node_path[i], node_path[i+1]))
                    edge_set.add(standardize_edge(node_path[-1], node_path[0]))
                    raw_pool.append((frozenset(edge_set), cost))
                except (ValueError, IndexError):
                    print(f"Warning: Format error on line {line_num+1}, skipped.")
                    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found", file=sys.stderr)
        return None
    print(f"--- Loading complete: {len(raw_pool)} original paths ---")
    return raw_pool

def filter_by_cost(raw_pool, c, op_cost):
    print(f"\n--- Step 2: Filtering by cost (c={c}) ---")
    if not raw_pool:
        print("Error: Original candidate pool is empty.")
        return []
    budget = c * op_cost * 1e8 
    print(f"Cost limit (c * optimal): {budget:.2f}")
    filtered_pool = []
    for edges, cost in raw_pool:
        if cost <= budget:
            filtered_pool.append((edges, cost))
    print(f"After cost filtering: {len(filtered_pool)} paths remaining (proceeding to Step 3)")
    return filtered_pool

def select_diverse_paths_from_pool(candidate_pool, k):
    print(f"\n--- Step 3: Running diversity selection (target k={k}) ---")
    candidate_list = list(candidate_pool)
    if not candidate_list:
        return [] 
    if len(candidate_list) < k:
        print(f"Warning: Candidate pool size ({len(candidate_list)}) is less than k ({k}).")
        k = len(candidate_list)

    selected_paths = [] 
    edge_counts = Counter() 
    try:
        random_start_index = random.randrange(len(candidate_list))
        S_1_tuple = candidate_list.pop(random_start_index)
        S_1_edges, S_1_cost = S_1_tuple
    except ValueError:
        return [] 
        
    selected_paths.append(S_1_tuple)
    edge_counts.update(S_1_edges) 
    print(f"Selected first tour (Random): cost = {S_1_cost:.2f}")

    for i in range(1, k):
        best_next_path_tuple = None
        best_next_path_index = -1
        lowest_score = float('inf')
        for j, (path_edges, path_cost) in enumerate(candidate_list):
            current_score = sum(edge_counts.get(edge, 0) for edge in path_edges)
            if current_score < lowest_score:
                lowest_score = current_score
                best_next_path_tuple = (path_edges, path_cost)
                best_next_path_index = j
        if best_next_path_tuple is not None:
            S_i_tuple = candidate_list.pop(best_next_path_index)
            S_i_edges, S_i_cost = S_i_tuple
            selected_paths.append(S_i_tuple)
            edge_counts.update(S_i_edges) 
        else:
            break
    print(f"\n--- Filtering complete: {len(selected_paths)} paths selected ---")
    return selected_paths


def analyze_jaccard_similarity(selected_paths_list):
    """
    Computes Jaccard similarity, returning the score list and statistical data.
    """
    path_edge_sets = [path[0] for path in selected_paths_list]

    
    empty_stats = {'avg': 0, 'max': 0, 'min': 0, 'std': 0, 'count': 0}

    if len(path_edge_sets) < 2:
        print("Fewer than 2 paths, cannot perform pairwise comparison.")
        return [], empty_stats

    jaccard_scores = []
    for set_A, set_B in itertools.combinations(path_edge_sets, 2):
        intersection_size = len(set_A.intersection(set_B))
        union_size = len(set_A.union(set_B))
        if union_size == 0:
            score = 1.0 
        else:
            score = intersection_size / union_size
        jaccard_scores.append(score)

    # calculate statistics
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    max_jaccard = max(jaccard_scores)
    min_jaccard = min(jaccard_scores)
    standard_error = np.std(jaccard_scores)
    
    print(f"\nAnalyzed {len(jaccard_scores)} combinations")
    print(f"  {'Average Jaccard Similarity :':<32} {avg_jaccard:.4f}")
    print(f"  {'Maximum Similarity :':<32} {max_jaccard:.4f}")
    print(f"  {'Minimum Similarity :':<32} {min_jaccard:.4f}")
    print(f"  {'Standard Deviation:':<32} {standard_error:.4f}")

    
    stats = {
        'avg': avg_jaccard,
        'max': max_jaccard,
        'min': min_jaccard,
        'std': standard_error,
        'count': len(jaccard_scores)
    }

    return jaccard_scores, stats  

def process_single_file(filepath, op_cost, k_val, c_val):

    print(f" file:       {filepath}")
    print(f" best cost:   {op_cost}")

    raw_pool = load_raw_solutions(filepath)
    if not raw_pool:
        print("Error: Failed to load candidate pool or file is empty.", file=sys.stderr)
        return [], [], None 

    filtered_pool = filter_by_cost(raw_pool, c_val, op_cost)
    if not filtered_pool:
        print("Error: No paths remaining after cost filtering.", file=sys.stderr)
        return [], [], None 

    selected_paths = select_diverse_paths_from_pool(set(filtered_pool), k_val)
    if not selected_paths:
        print("Error: Failed to select any paths.", file=sys.stderr)
        return [], [], None 
    
   
    print("\n--- (Individual Jaccard Analysis) ---")
    jaccard_scores, stats = analyze_jaccard_similarity(selected_paths)
    
    return selected_paths, jaccard_scores, stats

# =========================================================================
# MAIN 
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Select k low-cost and diverse paths from N TSP candidate pools.")
    parser.add_argument("-f", "--files", help="List of input files.", required=True, nargs='+') 
    parser.add_argument("-o", "--op_cost", help="Optimal cost/solution.", type=float, required=True)
    parser.add_argument("-k", "--k_val", help="Target number of paths (k).", type=int, default=30)
    parser.add_argument("-c", "--c_val", help="Cost multiplier (c).", type=float, default=2.0)
    
    args = parser.parse_args()

    print("=== TSP diversity filter ===")
    
    all_results_data = []
    all_results_labels = []
    
    output_txt = "jaccard_summary_results.txt"
    
    with open(output_txt, "a", encoding="utf-8") as f_out:
        
        header = f"{'Filename':<30} | {'Avg Jaccard':<12} | {'Max':<8} | {'Min':<8} | {'Std Dev':<8}\n"
        f_out.write(header)
        f_out.write("-" * 80 + "\n")
        
        print(f"Statistics will be written to: {output_txt}")

        for i in range(len(args.files)):
            filepath = args.files[i]
            op_cost = args.op_cost 
            
            print(f"\n--- [ Processing file {i+1}/{len(args.files)}: {filepath} ] ---")
            
            
            k_paths, jaccard_score, stats = process_single_file(filepath, op_cost, args.k_val, args.c_val)

            if k_paths and stats:
                all_results_data.append(jaccard_score)
                label_name = os.path.basename(filepath)
                all_results_labels.append(label_name)
                
                
                line = (f"{label_name:<30} | "
                        f"{stats['avg']:.4f}       | "
                        f"{stats['max']:.4f}   | "
                        f"{stats['min']:.4f}   | "
                        f"{stats['std']:.4f}\n")
                f_out.write(line)
                f_out.flush() 

                print(f"--- [ Processing complete, data written to txt ] ---")
            else:
                print(f"--- [ File {filepath} failed to process or yielded no results ] ---")
                f_out.write(f"{os.path.basename(filepath):<30} | FAILED or NO DATA\n")

    print("\n" + "=" * 40)
    print("All files processed.")
    print(f"Detailed statistics saved to: {output_txt}")

if __name__ == "__main__":
    main()
    