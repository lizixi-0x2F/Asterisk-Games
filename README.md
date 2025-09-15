# Asterisk Operator: ASPP Framework

This repository contains the implementation of the **Adjacency-Structured Parallel Propagation (ASPP)** model, a mathematical framework for abstract reasoning tasks described in the paper "Asterisk Operator: A Unified Framework for Adjacency-Structured Parallel Propagation in Abstract Reasoning".

## 📋 Paper Experiments Reproduction

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NetworkX
- Matplotlib
- tqdm

### Installation

```bash
pip install torch networkx matplotlib tqdm
```

## 🔬 Four Core Experiments

### 1. Graph 3-Coloring Ablation Study

**Purpose**: Validate ASPP architectural components using graph coloring problem

**Run the experiment**:
```bash
python graph_coloring_ablation.py
```

**Expected Results**:
- Generates ablation results showing architectural differences
- Produces visualizations in `graph_coloring_ablation_results.png`
- Saves detailed metrics in `ablation_results.json`

**Key Findings**:
- Edge weights improve accuracy by 2.6%
- Position encoding contributes 5.2% accuracy gain
- Unidirectional propagation shows better constraint satisfaction
- Learnable K-step converges faster than fixed steps

### 2. Conway's Game of Life - Turing Completeness Validation

**Purpose**: Demonstrate Turing completeness through cellular automaton simulation

**Run the experiment**:
```bash
python life_game_solver.py
```

**Expected Results**:
- Multi-step reasoning accuracy: ~99.8% circuit solving accuracy
- Validates universal computation capability
- Demonstrates 10-step Turing machine simulation

### 3. ARC2 Grid Games - Embedding-Asterisk Distillation

**⚠️ Critical Requirement**: This experiment requires pre-trained TreeGPT embeddings

**Setup TreeGPT First**:
```bash
git clone https://github.com/lizixi-0x2F/TreeGPT.git
cd TreeGPT
# Follow TreeGPT setup instructions to obtain pre-trained embeddings
```

**Then run distillation**:
```bash
python train_arc2_distillation.py
```

**Expected Results**:
- 100% validation accuracy on ARC2 with 6M parameters
- Knowledge transfer from TreeGPT prototype to ASPP framework
- Demonstrates embedding space reasoning capability

### 4. ARC2 Direct Training (From Scratch)

**Purpose**: Train ASPP model directly on ARC2 data without distillation

**Run the experiment**:
```bash
python train_arc2_on_scratch.py \
  --train_challenges arc-prize-2025/arc-agi_training_challenges.json \
  --train_solutions arc-prize-2025/arc-agi_training_solutions.json \
  --val_challenges arc-prize-2025/arc-agi_evaluation_challenges.json \
  --val_solutions arc-prize-2025/arc-agi_evaluation_solutions.json \
  --epochs 10 \
  --batch_size 1 \
  --hidden_dim 256
```

**Expected Results**:
- Direct graph reasoning on ARC2 tasks
- Knowledge graph construction with spatial, semantic, and logical relations
- Token-level accuracy metrics
- Demonstrates ASPP's capability for direct abstract reasoning

**Key Features**:
- **Knowledge Embedding Layer**: Single-layer token embedding for graph nodes
- **Graph Construction**: Builds knowledge graphs with spatial adjacency, semantic similarity, and logical dependency relations
- **Direct Reasoning**: Applies ASPP operator directly on constructed knowledge graphs
- **Token-level Training**: Cross-entropy loss on token predictions with padding handling

## 📊 Results Summary

| Experiment | Accuracy | Parameters | Key Contribution |
|------------|----------|------------|------------------|
| Graph 3-Coloring | 31.7-38.2% | 6M | Architectural ablation |
| Conway's Game of Life | 99.8% | - | Turing completeness |
| ARC2 via Distillation | 100% | 6M | Knowledge transfer |
| ARC2 Direct Training | Token-level metrics | 6M | Direct graph reasoning |

## 🏗️ Project Structure

```
.
├── aspp_model/           # Core ASPP implementation
│   ├── aspp_operator.py  # ASPP operator (Def 1.3-1.4)
│   ├── ffn_update.py     # FFN-based local update rules
│   ├── graph_structure.py # Graph data structures
│   └── rope_encoding.py  # Rotary Position Encoding
├── graph_coloring_ablation.py  # Experiment 1
├── life_game_solver.py         # Experiment 2
├── train_arc2_distillation.py  # Experiment 3 (requires TreeGPT)
├── train_arc2_on_scratch.py    # Experiment 4 (direct training)
└── package_asterisk/
    └── arXiv_paper.tex   # Main paper
```

## 🎯 Key Mathematical Concepts Implemented

- **Definition 1.3**: ASPP Operator `Φ(H^(t); E) = H^(t+1)`
- **Definition 1.4**: K-Step Reasoning Evolution `Φ^(K)(H^(0); E)`
- **Theorem 2.1**: Universality (MPNN simulation)
- **Theorem 2.2**: Convergence under contraction mapping

## 📈 Performance Highlights

- **Graph 3-Coloring**: Clear architectural differentiation (31.7-38.2% accuracy)
- **Conway's Game of Life**: 99.8% multi-step reasoning accuracy
- **ARC2 via Distillation**: 100% accuracy with only 6M parameters
- **Training Efficiency**: Converges within 1500 steps (TreeGPT prototype)

## 🔧 Customization

The framework supports:
- Dynamic graph structure switching
- Configurable update rules (FFN-based)
- Learnable reasoning depth (K-step parameterization)
- Bidirectional/Unidirectional propagation
- Edge weight adaptation

## 📝 Citation

If you use this work, please cite:
```
@article{li2025asterisk,
  title={Asterisk Operator: A Unified Framework for Adjacency-Structured Parallel Propagation in Abstract Reasoning},
  author={Li, Zixi},
  journal={arXiv preprint},
  year={2025}
}
```

## 🚀 Future Work

- Adaptive graph construction methods
- Non-contraction dynamics extension
- Large-scale applications in robotics and planning
- Integration with other neural-symbolic approaches

## 📞 Contact

For questions about the implementation or reproduction:
- Email: lizx93@mail2.sysu.edu.cn
- GitHub: https://github.com/lizixi-0x2F/TreeGPT

---

**Note**: The ARC2 distillation experiment requires pre-trained TreeGPT embeddings. Ensure you clone and setup the TreeGPT repository first before attempting to reproduce the 100% accuracy results.