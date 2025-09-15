"""
FFN-based Local Update Rule Implementation
严格遵循论文定义 1.3: 局部更新规则 φ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Any

class SwiGLU(nn.Module):
    """SwiGLU激活函数：结合Swish和GLU门控"""
    def __init__(self, dim: int):
        super().__init__()
        # 标准SwiGLU实现：输入dim，输出dim/2
        self.w1 = nn.Linear(dim, dim, bias=False)
        self.w2 = nn.Linear(dim, dim, bias=False)
        self.w3 = nn.Linear(dim, dim // 2, bias=False)

    def forward(self, x):
        # 标准SwiGLU公式：Swish(w1*x) ⊗ w2*x
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.w3(F.silu(x1) * x2)

class FFNUpdate(nn.Module):
    """
    FFN局部更新规则实现
    定义 1.3: h_i^{(t+1)} = φ(h_i^{(t)}, {h_j^{(t)} | (v_j, v_i) ∈ E})
    
    使用前馈神经网络实现局部更新规则
    """
    
    def __init__(self, hidden_dim: int, contraction_factor: Optional[float] = None):
        """
        初始化FFN局部更新规则

        Args:
            hidden_dim: 隐藏层维度
            contraction_factor: 压缩因子（用于定理2.3的收敛性验证）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.contraction_factor = contraction_factor

        # FFN架构：输入 = 当前节点状态 + 邻居状态聚合
        # 邻居状态通过求和聚合（类似MPNN）
        input_dim = hidden_dim * 2  # 当前状态 + 聚合邻居状态

        # 使用SwiGLU架构（增强正则化）
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 4),
            SwiGLU(hidden_dim * 4),  # 输入hidden_dim*4，输出hidden_dim*2
            nn.Dropout(0.3),  # 增加dropout率
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # 中间层dropout
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)  # 输出层dropout
        )

        # 边投影机制（将节点和邻居的边信息投影）
        self.edge_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入：当前节点状态 + 邻居状态
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)  # 边投影dropout
        )

        # 门控聚合机制（学习邻居聚合的重要性）
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入：当前状态 + 邻居聚合
            nn.Sigmoid(),  # 输出0-1之间的门控权重
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)  # 门控dropout
        )

        # 添加权重衰减正则化（通过初始化控制）
        self._initialize_with_regularization()

    def _initialize_with_regularization(self):
        """使用正则化友好的初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 使用较小的初始化方差，促进泛化
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.8)  # 较小的增益
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def apply_gradient_regularization(self, weight_decay: float = 1e-4):
        """应用梯度正则化（L2正则化）"""
        for param in self.parameters():
            if param.grad is not None and param.requires_grad:
                param.grad.data.add_(weight_decay * param.data)


    def forward(self, current_state: torch.Tensor, neighbor_states: List[torch.Tensor]) -> torch.Tensor:
        """
        局部更新规则前向传播（带残差连接和门控）

        Args:
            current_state: 当前节点状态 h_i^(t) ∈ R^d
            neighbor_states: 邻居状态列表 [{h_j^(t) | (v_j, v_i) ∈ E}]

        Returns:
            更新后的节点状态
        """

        # 聚合邻居状态（使用求和，类似MPNN）
        if neighbor_states:
            # 将所有邻居状态堆叠并求和
            neighbor_agg = torch.stack(neighbor_states).sum(dim=0)
        else:
            # 如果没有邻居，使用零向量
            neighbor_agg = torch.zeros_like(current_state)

        # 拼接当前状态和聚合的邻居状态
        combined = torch.cat([current_state, neighbor_agg], dim=-1)

        # 通过FFN进行更新（带残差连接）
        # 残差连接：updated_state = ffn(combined) + current_state
        fnn_output = self.ffn(combined)
        ffn_updated_state = fnn_output + current_state  # 残差连接

        return ffn_updated_state
    
    def vectorized_forward(self, H: torch.Tensor, neighbor_aggregates: torch.Tensor) -> torch.Tensor:
        """
        向量化前向传播（带门控聚合批量处理所有节点）

        Args:
            H: 当前状态配置 [n_nodes, hidden_dim]
            neighbor_aggregates: 邻居聚合状态 [n_nodes, hidden_dim]

        Returns:
            更新后的状态 [n_nodes, hidden_dim]
        """
        # 计算门控权重（学习邻居聚合的重要性）
        gate_input = torch.cat([H, neighbor_aggregates], dim=-1)
        gate_weights = self.gate(gate_input)  # [n_nodes, hidden_dim]

        # 应用门控到邻居聚合
        gated_neighbors = neighbor_aggregates * gate_weights

        # 拼接当前状态和门控后的邻居状态
        combined = torch.cat([H, gated_neighbors], dim=-1)

        # 通过FFN进行更新（带残差连接）
        fnn_output = self.ffn(combined)
        ffn_updated_state = fnn_output + H  # 残差连接

        return ffn_updated_state

    def vectorized_forward_with_edges(self, H: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        向量化前向传播（带边投影）

        Args:
            H: 当前状态配置 [n_nodes, hidden_dim]
            edge_index: 边索引张量 [2, E]

        Returns:
            更新后的状态 [n_nodes, hidden_dim]
        """
        src_nodes = edge_index[0]  # 源节点索引
        tgt_nodes = edge_index[1]  # 目标节点索引

        # 计算每条边的消息：msg = W_edge([h_src, h_tgt])
        edge_states = torch.cat([H[src_nodes], H[tgt_nodes]], dim=-1)
        edge_messages = self.edge_proj(edge_states)  # [E, hidden_dim]

        # 聚合边消息到目标节点
        neighbor_aggregates = torch.zeros_like(H)
        neighbor_aggregates.index_add_(0, tgt_nodes, edge_messages)

        # 计算门控权重
        gate_input = torch.cat([H, neighbor_aggregates], dim=-1)
        gate_weights = self.gate(gate_input)  # [n_nodes, hidden_dim]

        # 应用门控到邻居聚合
        gated_neighbors = neighbor_aggregates * gate_weights

        # 拼接当前状态和门控后的邻居状态
        combined = torch.cat([H, gated_neighbors], dim=-1)

        # 通过FFN进行更新（带残差连接）
        fnn_output = self.ffn(combined)
        ffn_updated_state = fnn_output + H  # 残差连接

        return ffn_updated_state
    
    
    def check_contraction_property(self, state1: torch.Tensor, neighbors1: List[torch.Tensor],
                                 state2: torch.Tensor, neighbors2: List[torch.Tensor]) -> float:
        """
        检查压缩映射性质（用于定理2.3验证）
        
        Returns:
            压缩系数 c
        """
        updated1 = self.forward(state1, neighbors1)
        updated2 = self.forward(state2, neighbors2)
        
        diff_output = torch.norm(updated1 - updated2)
        max_input_diff = max(
            torch.norm(state1 - state2),
            max(torch.norm(n1 - n2) for n1, n2 in zip(neighbors1, neighbors2)) if neighbors1 else 0
        )
        
        if max_input_diff > 0:
            contraction_ratio = diff_output / max_input_diff
            return contraction_ratio.item()
        return 0.0
