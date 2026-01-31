# Generating Diverse TSP Tours via a Combination of Graph Pointer Network and Dispersion

This repository is the official implementation of our research paper:

> **Generating Diverse TSP Tours via a Combination of Graph Pointer Network and Dispersion**
> Hao-Tsung Yang, Ssu-Yuan Lo, Kuan-Lun Chen, Ching-Kai Wang.
> *National Central University (NCU), Taiwan.*

---> [arXiv](https://arxiv.org/abs/2601.01132) 

##  Contact

> htyang@ncu.edu.tw

##  Introduction
This project proposes a novel solver capable of generating diverse and high-quality solutions for the Traveling Salesperson Problem (TSP). By integrating **Graph Pointer Networks (GPN)** with a **Dispersion Algorithm**, our method strikes a favorable balance between solution quality and diversity.

Key components of our approach:

- **Graph Pointer Network (GPN)**: Serves as the backbone to learn the structural patterns of TSP instances and generate initial high-quality tours.
- **Dispersion Algorithm**: A post-processing mechanism designed to strictly enforce diversity, ensuring the generated tours are distinct from one another while maintaining competitive lengths.

##  Dependencies

```bash
python >= 3.10
pytorch >= 2.0.0
numpy
scipy
```

## Usage
You can train the Spanning Tree and Perfect Matching models separately using the commands below.
```bash
python ./GPN_spanning_tree/train.py
python ./GPN_perfect_matching/train.py
```

To generate TSP solutions using the pre-trained models, execute the following command from the project root directory:

```bash
python ./inference/main.py

For generating the results of k valid tours,
run: python inference/filter.py 
    -f tour path solution file name ex."Bi-Criteria_Heuristic/Results/criteria_berlin52.solution" "inference/Result/berlin52" "NMA/Results/berlin52"
    -o optimal_cost of the graph  "berlin52 = 4.3345  eil101 = 8.1688   rd400 = 15.3190  rat783 = 15.1828"
    -k the numbers of chosen valid tour  
    -c a cost multiplier for limitation
```


##  Citations

If you find our work useful, please cite our paper:

```bibtex
@misc{yang2026generating,
      title={Generating Diverse TSP Tours via a Combination of Graph Pointer Network and Dispersion}, 
      author={Hao-Tsung Yang and Ssu-Yuan Lo and Kuan-Lun Chen and Ching-Kai Wang},
      year={2026},
      eprint={2601.01132},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={[https://arxiv.org/abs/2601.01132](https://arxiv.org/abs/2601.01132)}, 
}
```

##  Acknowledgements

We thank the authors of the following repositories for providing their code, which greatly facilitated our research.


### Code Base
* Our implementation is primarily based on [Graph Pointer Network](https://github.com/qiang-ma/graph-pointer-network)

### Baselines
We also acknowledge the following open-source projects used as baselines in our experiments:
* **RF-MA3S:** [github](https://github.com/LiQisResearch/KDD--RF-MA3S)
* **NMA**: [github](https://github.com/GnauhGnit/MSTSP)


