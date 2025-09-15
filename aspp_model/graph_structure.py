"""
严格遵循论文定义 1.1: 广义推理结构
"""

import torch
from typing import List, Tuple

class GraphStructure:
    """
    广义推理结构 (Generalized Reasoning Structure)
    定义 1.1: \mathcal{G} = (V, E, \mathcal{H})
    使用直接边存储，提高性能并保持与ASPPOperator兼容
    """
    
    def __init__(self, n_nodes: int, hidden_dim: int, device: torch.device = torch.device('cpu')):
        """
        初始化广义推理结构
        
        Args:
            n_nodes: 节点数量 |V| = n
            hidden_dim: 状态空间维度 d (通常 \mathcal{H} = \mathbb{R}^d)
            device: 计算设备
        """
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 直接边存储（PyG风格）
        self.edge_index = None  # [2, E] 张量，第一行为源节点，第二行为目标节点
        self.edge_count = 0
        
        # 状态空间 \mathcal{H} = \mathbb{R}^d
        self.state_space = hidden_dim
        
    def build_adjacency_list(self, edges: List[Tuple[int, int]]):
        """
        构建直接边存储（PyG风格）
        
        Args:
            edges: 边列表 [(source, target), ...]
        """
        n_edges = len(edges)
        self.edge_count = n_edges
        
        # 构建直接边索引 [2, E]
        if n_edges > 0:
            sources, targets = zip(*edges)
            self.edge_index = torch.tensor([sources, targets], dtype=torch.long, device=self.device)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
    
    def build_linear_adjacency(self):
        """
        构建线性相邻关系 E_adj = {(v_i, v_{i+1}) | i=1,...,n-1}
        用于定理 2.2 的计算效率验证
        """
        edges = []
        for i in range(self.n_nodes - 1):
            edges.append((i, i + 1))  # 单向相邻关系
        self.build_adjacency_list(edges)
    
    def get_neighbors(self, node_idx: int) -> List[int]:
        """
        获取指定节点的所有邻居节点
        
        Args:
            node_idx: 节点索引
            
        Returns:
            邻居节点索引列表
        """
        if self.edge_index is None:
            return []
        
        # 从edge_index中查找所有目标节点为node_idx的边
        neighbors = []
        sources = self.edge_index[0]
        targets = self.edge_index[1]
        
        # 查找所有源节点为node_idx的边
        mask = sources == node_idx
        neighbors.extend(targets[mask].tolist())
        
        return neighbors
    
    def get_incoming_edges(self, node_idx: int) -> List[int]:
        """
        获取指向指定节点的所有边对应的源节点
        用于ASPP更新规则: {h_j | (v_j, v_i) ∈ E}
        
        Args:
            node_idx: 目标节点索引
            
        Returns:
            源节点索引列表
        """
        if self.edge_index is None:
            return []
        
        # 从edge_index中查找所有目标节点为node_idx的边
        sources = self.edge_index[0]
        targets = self.edge_index[1]
        
        # 查找所有目标节点为node_idx的边
        mask = targets == node_idx
        return sources[mask].tolist()
    
    def build_incoming_adjacency_matrix(self) -> torch.Tensor:
        """
        构建入边邻接矩阵（向量化版本）
        
        Returns:
            入边邻接矩阵，形状为 [n_nodes, n_nodes]
        """
        n_nodes = self.n_nodes
        adj_matrix = torch.zeros((n_nodes, n_nodes), device=self.device)
        
        if self.edge_index is not None:
            # 直接从edge_index构建邻接矩阵
            sources = self.edge_index[0]
            targets = self.edge_index[1]
            adj_matrix[targets, sources] = 1
        
        return adj_matrix
    
    def get_all_edges(self) -> List[Tuple[int, int]]:
        """
        获取所有边的列表
        
        Returns:
            所有边的列表 [(source, target), ...]
        """
        if self.edge_index is None:
            return []
        
        # 直接从edge_index转换为边列表
        sources = self.edge_index[0].tolist()
        targets = self.edge_index[1].tolist()
        return list(zip(sources, targets))
    
    def validate_structure(self):
        """验证图结构完整性"""
        assert self.edge_index is not None, "Edge index not initialized"
        assert self.edge_count == self.edge_index.shape[1], f"Edge count mismatch: {self.edge_count} != {self.edge_index.shape[1]}"
        
        # 检查所有边是否有效
        if self.edge_count > 0:
            sources = self.edge_index[0]
            targets = self.edge_index[1]
            assert torch.all((sources >= 0) & (sources < self.n_nodes)), "Invalid source nodes"
            assert torch.all((targets >= 0) & (targets < self.n_nodes)), "Invalid target nodes"
        
        print(f"Graph structure validated: {self.n_nodes} nodes, {self.edge_count} edges")
