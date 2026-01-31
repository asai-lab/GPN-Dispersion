# Generating Diverse TSP Tours via a Combination of Graph Pointer Network and Dispersion

This repository is the official implementation of our research paper:

> **Generating Diverse TSP Tours via a Combination of Graph Pointer Network and Dispersion**
> Hao-Hsung Yang, Ssu-Yuan Lo, Kuan-Lun Chen, Ching-Kai Wang.
> *National Central University (NCU), Taiwan.*

---> [arXiv](https://arxiv.org/abs/2601.01132) 

## ✉️ Contact

> htyang@ncu.edu.tw

## 📝 Introduction
This project proposes a novel solver capable of generating diverse and high-quality solutions for the Traveling Salesperson Problem (TSP). By integrating **Graph Pointer Networks (GPN)** with a **Dispersion Algorithm**, our method strikes a favorable balance between solution quality and diversity.

Key components of our approach:

- **Graph Pointer Network (GPN)**: Serves as the backbone to learn the structural patterns of TSP instances and generate initial high-quality tours.
- **Dispersion Algorithm**: A post-processing mechanism designed to strictly enforce diversity, ensuring the generated tours are distinct from one another while maintaining competitive lengths.

![Method Architecture](fig/architecture.png) 
*(If you have a diagram of your model, put it in a 'fig' folder and link it here)*

## ⚙️ Dependencies

```bash
python >= 3.10
pytorch >= 2.0.0
numpy
scipy
# Add other libraries you used




If you find our work useful, please cite our paper:

```bibtex
@misc{yang2026generatingdiversetsptours,
      title={Generating Diverse TSP Tours via a Combination of Graph Pointer Network and Dispersion}, 
      author={Hao-Tsung Yang and Ssu-Yuan Lo and Kuan-Lun Chen and Ching-Kai Wang},
      year={2026},
      eprint={2601.01132},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2601.01132}, 
}
```

