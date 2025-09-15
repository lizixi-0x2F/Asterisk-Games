#!/usr/bin/env python3
"""
Graph 3-Coloring Ablation Study for ASPP Framework

This script implements a comprehensive ablation study for the Asterisk Operator
framework using the Graph 3-Coloring problem as a testbed.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import your ASPP components
from aspp_model.aspp_operator import ASPPOperator
from aspp_model.ffn_update import FFNUpdate
from aspp_model.graph_structure import GraphStructure


def generate_3coloring_dataset(num_graphs: int = 100, min_nodes: int = 10, max_nodes: int = 30):
    """Generate a dataset of 3-colorable graphs with known solutions."""
    graphs = []
    solutions = []

    for _ in range(num_graphs):
        n_nodes = np.random.randint(min_nodes, max_nodes + 1)

        # Generate a 3-colorable graph (bipartite graphs are 2-colorable,
        # so we use tripartite for 3-coloring)
        while True:
            G = nx.erdos_renyi_graph(n_nodes, 0.3)
            if nx.is_connected(G):
                try:
                    # Try to 3-color it
                    coloring = nx.greedy_color(G, strategy='largest_first', interchange=True)
                    if max(coloring.values()) < 3:  # Ensure it's 3-colorable
                        graphs.append(G)
                        solutions.append(coloring)
                        break
                except:
                    continue

    return graphs, solutions


def graph_to_tensor(graph: nx.Graph) -> Tuple[list, torch.Tensor]:
    """Convert networkx graph to edge list and node features."""
    n_nodes = len(graph.nodes())

    # Get edge list
    edges = list(graph.edges())

    # Node features: degree, clustering coefficient, etc.
    degrees = np.array([d for n, d in graph.degree()])
    clustering = np.array(list(nx.clustering(graph).values()))

    node_features = np.column_stack([degrees, clustering])

    return edges, torch.tensor(node_features, dtype=torch.float32)


class ColoringASPP(nn.Module):
    """ASPP model for graph 3-coloring problem."""

    def __init__(self, hidden_dim: int = 64, use_edge_weights: bool = True,
                 use_position_encoding: bool = True, bidirectional: bool = True,
                 learnable_k: bool = True, max_steps: int = 20):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_edge_weights = use_edge_weights
        self.use_position_encoding = use_position_encoding
        self.bidirectional = bidirectional
        self.learnable_k = learnable_k
        self.max_steps = max_steps

        # Input projection
        self.input_proj = nn.Linear(2, hidden_dim)

        # Position encoding (if used)
        if use_position_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, hidden_dim))

        # ASPP update rule
        self.update_rule = FFNUpdate(hidden_dim)

        # Learnable K parameter
        if learnable_k:
            self.k_param = nn.Parameter(torch.tensor(0.5))
        else:
            self.k_param = None

        # Output projection to 3 colors
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, node_features: torch.Tensor, graph_edges: list) -> Tuple[torch.Tensor, int]:
        """Forward pass for graph coloring."""
        n_nodes = node_features.shape[0]

        # Project input features
        h = self.input_proj(node_features)

        # Add position encoding if enabled
        if self.use_position_encoding:
            h = h + self.pos_encoding

        # Determine number of steps
        if self.learnable_k and self.k_param is not None:
            k_steps = int(torch.sigmoid(self.k_param) * self.max_steps) + 1
        else:
            k_steps = self.max_steps

        # Create edge index tensor
        if graph_edges:
            sources, targets = zip(*graph_edges)
            edge_index = torch.tensor([sources, targets], dtype=torch.long, device=h.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=h.device)

        # ASPP operator
        aspp_op = ASPPOperator(self.update_rule, self.hidden_dim)
        aspp_op.set_graph(n_nodes, edge_index)

        # Apply ASPP for K steps (no batch dimension)
        current_states = h  # Shape: [n_nodes, hidden_dim]
        for step in range(k_steps):
            current_states = aspp_op(current_states)

        # Predict colors
        final_states = current_states  # Shape: [n_nodes, hidden_dim]
        color_logits = self.output_proj(final_states)

        return color_logits, k_steps


def compute_coloring_metrics(predictions: torch.Tensor, targets: Dict[int, int],
                            adjacency: torch.Tensor) -> Dict[str, float]:
    """Compute metrics for graph coloring performance."""

    # Convert predictions to hard assignments
    pred_colors = torch.argmax(predictions, dim=-1)

    # Accuracy
    correct = 0
    total = len(targets)
    for node, true_color in targets.items():
        if pred_colors[node].item() == true_color:
            correct += 1
    accuracy = correct / total

    # Constraint violation rate
    violations = 0
    n_edges = 0

    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i, j] > 0 and i != j:  # Edge exists
                n_edges += 1
                if pred_colors[i] == pred_colors[j]:
                    violations += 1

    violation_rate = violations / n_edges if n_edges > 0 else 0

    return {
        'accuracy': accuracy,
        'violation_rate': violation_rate,
        'total_violations': violations
    }


def run_ablation_study():
    """Run comprehensive ablation study for Graph 3-Coloring."""

    # Generate test dataset
    print("Generating 3-coloring dataset...")
    test_graphs, test_solutions = generate_3coloring_dataset(50, 10, 20)

    # Ablation configurations
    ablation_configs = [
        {'use_edge_weights': True, 'use_position_encoding': True, 'bidirectional': True, 'learnable_k': True, 'name': 'Full Model'},
        {'use_edge_weights': False, 'use_position_encoding': True, 'bidirectional': True, 'learnable_k': True, 'name': 'No Edge Weights'},
        {'use_edge_weights': True, 'use_position_encoding': False, 'bidirectional': True, 'learnable_k': True, 'name': 'No Position Encoding'},
        {'use_edge_weights': True, 'use_position_encoding': True, 'bidirectional': False, 'learnable_k': True, 'name': 'Unidirectional Only'},
        {'use_edge_weights': True, 'use_position_encoding': True, 'bidirectional': True, 'learnable_k': False, 'name': 'Fixed K=10'},
        {'use_edge_weights': False, 'use_position_encoding': False, 'bidirectional': False, 'learnable_k': False, 'name': 'Minimal Configuration'}
    ]

    results = {}

    for config in tqdm(ablation_configs, desc="Running ablation study"):
        config_name = config['name']
        results[config_name] = {'accuracy': [], 'violation_rate': [], 'convergence_steps': [], 'total_violations': []}

        model = ColoringASPP(
            hidden_dim=64,
            use_edge_weights=config['use_edge_weights'],
            use_position_encoding=config['use_position_encoding'],
            bidirectional=config['bidirectional'],
            learnable_k=config['learnable_k']
        )

        # Test on each graph
        for graph, solution in zip(test_graphs, test_solutions):
            edges, node_features = graph_to_tensor(graph)

            # Forward pass
            with torch.no_grad():
                predictions, k_steps = model(node_features, edges)

            # Compute metrics (need adjacency matrix for constraint checking)
            adj_matrix = nx.adjacency_matrix(graph).toarray()
            metrics = compute_coloring_metrics(predictions, solution, torch.tensor(adj_matrix, dtype=torch.float32))

            results[config_name]['accuracy'].append(metrics['accuracy'])
            results[config_name]['violation_rate'].append(metrics['violation_rate'])
            results[config_name]['convergence_steps'].append(k_steps)
            results[config_name]['total_violations'].append(metrics['total_violations'])

    return results


def analyze_and_visualize_results(results: Dict):
    """Analyze and visualize ablation study results."""

    # Convert to DataFrame-like structure for analysis
    config_names = list(results.keys())
    metrics = ['accuracy', 'violation_rate', 'convergence_steps']

    # Create summary statistics
    summary = {}
    for config in config_names:
        summary[config] = {}
        for metric in metrics:
            values = results[config][metric]
            summary[config][f'{metric}_mean'] = np.mean(values)
            summary[config][f'{metric}_std'] = np.std(values)

    # Print results
    print("\n" + "="*80)
    print("GRAPH 3-COLORING ABLATION STUDY RESULTS")
    print("="*80)

    for config in config_names:
        print(f"\n{config}:")
        print(f"  Accuracy: {summary[config]['accuracy_mean']:.3f} ± {summary[config]['accuracy_std']:.3f}")
        print(f"  Violation Rate: {summary[config]['violation_rate_mean']:.3f} ± {summary[config]['violation_rate_std']:.3f}")
        print(f"  Convergence Steps: {summary[config]['convergence_steps_mean']:.1f} ± {summary[config]['convergence_steps_std']:.1f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy comparison
    acc_means = [summary[config]['accuracy_mean'] for config in config_names]
    acc_stds = [summary[config]['accuracy_std'] for config in config_names]

    axes[0, 0].bar(config_names, acc_means, yerr=acc_stds, alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Violation rate comparison
    viol_means = [summary[config]['violation_rate_mean'] for config in config_names]
    viol_stds = [summary[config]['violation_rate_std'] for config in config_names]

    axes[0, 1].bar(config_names, viol_means, yerr=viol_stds, alpha=0.7, color='red')
    axes[0, 1].set_title('Constraint Violation Rate')
    axes[0, 1].set_ylabel('Violation Rate')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Convergence steps
    steps_means = [summary[config]['convergence_steps_mean'] for config in config_names]
    steps_stds = [summary[config]['convergence_steps_std'] for config in config_names]

    axes[1, 0].bar(config_names, steps_means, yerr=steps_stds, alpha=0.7, color='green')
    axes[1, 0].set_title('Average Convergence Steps')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Component importance analysis
    component_impact = {
        'Edge Weights': summary['Full Model']['accuracy_mean'] - summary['No Edge Weights']['accuracy_mean'],
        'Position Encoding': summary['Full Model']['accuracy_mean'] - summary['No Position Encoding']['accuracy_mean'],
        'Bidirectional': summary['Full Model']['accuracy_mean'] - summary['Unidirectional Only']['accuracy_mean'],
        'Learnable K': summary['Full Model']['accuracy_mean'] - summary['Fixed K=10']['accuracy_mean']
    }

    axes[1, 1].bar(list(component_impact.keys()), list(component_impact.values()), alpha=0.7, color='purple')
    axes[1, 1].set_title('Component Importance (Accuracy Impact)')
    axes[1, 1].set_ylabel('Accuracy Difference from Full Model')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('graph_coloring_ablation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save results to JSON
    with open('ablation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("Starting Graph 3-Coloring Ablation Study...")

    # Run ablation study
    results = run_ablation_study()

    # Analyze and visualize results
    summary = analyze_and_visualize_results(results)

    print("\nAblation study completed! Results saved to:")
    print("- ablation_results.json")
    print("- graph_coloring_ablation_results.png")