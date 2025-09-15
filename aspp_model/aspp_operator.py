"""
ASPP Operator Implementation
严格遵循论文定义 1.3 和 1.4: ASPP算子和K步推理演化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from .ffn_update import FFNUpdate

class ASPPOperator(nn.Module):
    """
    ASPP算子实现（支持动态图结构和并行生成）
    定义 1.3: H^{(t+1)} = Φ(H^{(t)}; E)
    定义 1.4: ̂H = Φ^{(K)}(H^{(0)}; E)
    新增：并行图生成能力

    """
    
    def __init__(self, update_rule: FFNUpdate, hidden_dim: int = 256, device: torch.device = None,
                instance_id: str = "default"):
        """
        初始化ASPP算子（支持动态图结构、双向聚合、并行生成和多实例）
        默认使用双向邻居聚合
        新增并行图生成能力和多层算子实例管理

        Args:
            update_rule: 局部更新规则 φ
            hidden_dim: 隐藏层维度
            device: 计算设备
            instance_id: 算子实例标识符，用于多实例管理
        """
        super().__init__()
        self.update_rule = update_rule
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cpu')

        self.instance_id = instance_id

        # 可学习K步参数
        self.learnable_K = nn.Parameter(torch.tensor(15.0), requires_grad=True)  # 初始K=10
        self.max_k_step = 15  # 最大K步=15

        # 边权重自适应（将在运行时根据具体图结构创建）
        self.edge_weights = None
        self.current_graph = None

        # 算子实例管理
        self.child_operators = nn.ModuleDict()
        self.operator_stack = []

        # 并行生成组件（可选）
        self.has_parallel_generation = False
        self._init_parallel_generation_components()
    
    def _init_parallel_generation_components(self):
        """初始化并行图生成组件 - 简化版本"""
        # 完整的token词汇表: 0-9 颜色 + 10 分隔符 + 11 填充符
        self.vocab_size = 12  # 0-9: 颜色, 10: 行分隔符, 11: 填充符

        # token定义
        self.ROW_SEP = 10     # 行分隔符（用于网格还原）
        self.PAD_TOKEN = 11   # 填充符（在损失计算中忽略）

        # 输出投影层
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)

        # 位置编码
        self.position_emb = nn.Embedding(8192, self.hidden_dim)

        self.has_parallel_generation = True
    
    def set_graph(self, n_nodes: int, edge_index: torch.LongTensor):
        """
        设置当前图结构（使用TreeFFN风格的直接边存储）
        
        Args:
            n_nodes: 节点数量
            edge_index: 边索引张量 [2, E]，第一行为源节点，第二行为目标节点
        """
        self.n_nodes = n_nodes
        self.edge_index = edge_index
        self.edge_count = edge_index.shape[1]
        
        # 为当前图结构创建边权重参数
        if self.edge_count > 0:
            self.edge_weights = nn.Parameter(
                torch.ones(self.edge_count, device=self.device),
                requires_grad=True
            )
        else:
            self.edge_weights = None
        
    def forward(self, H: torch.Tensor, K: int = None, positions: torch.Tensor = None,
                use_stack: bool = False) -> torch.Tensor:
        """
        K步推理演化（使用可学习K步参数，支持算子栈）

        Args:
            H: 初始状态配置 H^{(0)} ∈ \mathbb{R}^{n × d}
            K: 演化步数（如果为None则使用可学习参数）
            positions: 位置索引，用于旋转位置编码
            use_stack: 是否使用算子栈进行多层处理

        Returns:
            最终状态配置
        """
        assert hasattr(self, 'n_nodes'), "Graph not set. Call set_graph() first."
        assert H.shape[0] == self.n_nodes, f"State dimension mismatch: {H.shape[0]} != {self.n_nodes}"
        assert H.shape[1] == self.hidden_dim, f"State dimension mismatch: {H.shape[1]} != {self.hidden_dim}"

        current_H = H.clone()

        # 如果没有提供位置，使用节点索引作为位置
        if positions is None:
            positions = torch.arange(self.n_nodes, device=self.device)

        # 使用可学习K步参数（sigmoid映射到1-max_k_step范围）
        if K is None:
            K_float = torch.sigmoid(self.learnable_K) * (self.max_k_step - 1) + 1  # 1-max_k_step范围
            K = int(torch.round(K_float).item())

        if use_stack and self.operator_stack:
            # 使用算子栈进行多层处理
            return self._apply_operator_stack(current_H, K, positions)
        else:
            # 单算子K步演化
            for _ in range(K):
                current_H = self.single_step(current_H, positions)

        return current_H
    
    def get_learnable_k_value(self) -> Dict[str, Any]:
        """获取可学习K步参数的值"""
        # 使用sigmoid映射到1-max_k_step范围
        K_float = torch.sigmoid(self.learnable_K) * (self.max_k_step - 1) + 1
        K_int = int(torch.round(K_float).item())

        return {
            'raw_value': self.learnable_K.detach().item(),
            'float_value': K_float.detach().item(),
            'int_value': K_int,
            'max_k_step': self.max_k_step,
            'requires_grad': self.learnable_K.requires_grad
        }

    def create_child_operator(self, instance_id: str, update_rule: FFNUpdate = None,
                            hidden_dim: int = None, **kwargs) -> 'ASPPOperator':
        """
        创建子算子实例

        Args:
            instance_id: 子算子标识符
            update_rule: 更新规则（如果为None则复制当前规则）
            hidden_dim: 隐藏层维度（如果为None则使用当前维度）
            **kwargs: 其他初始化参数

        Returns:
            新创建的ASPP算子实例
        """
        if instance_id in self.child_operators:
            raise ValueError(f"Operator instance '{instance_id}' already exists")

        # 使用提供的参数或默认值
        update_rule = update_rule or self.update_rule
        hidden_dim = hidden_dim or self.hidden_dim

        # 创建新算子实例
        child_op = ASPPOperator(
            update_rule=update_rule,
            hidden_dim=hidden_dim,
            device=self.device,
            instance_id=instance_id,
            **kwargs
        )

        # 添加到子算子字典
        self.child_operators[instance_id] = child_op

        return child_op

    def add_to_stack(self, operator_instance: 'ASPPOperator'):
        """
        将算子实例添加到算子栈

        Args:
            operator_instance: ASPP算子实例
        """
        self.operator_stack.append(operator_instance)

    def clear_stack(self):
        """清空算子栈"""
        self.operator_stack.clear()

    def _apply_operator_stack(self, H: torch.Tensor, K: int, positions: torch.Tensor) -> torch.Tensor:
        """
        应用算子栈进行多层处理

        Args:
            H: 初始状态
            K: 每层步数
            positions: 位置索引

        Returns:
            多层处理后的状态
        """
        current_H = H.clone()

        for operator in self.operator_stack:
            # 确保每个算子使用相同的图结构
            if hasattr(self, 'n_nodes') and hasattr(self, 'edge_index'):
                operator.set_graph(self.n_nodes, self.edge_index)
            current_H = operator.forward(current_H, K, positions, use_stack=False)

        return current_H

    def single_step(self, H: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        单步ASPP更新 - PyTorch向量化版本
        
        Args:
            H: 当前状态配置 H^{(t)} ∈ \mathbb{R}^{n × d}
            positions: 位置索引，用于旋转位置编码
            
        Returns:
            更新后的状态配置
        """
        # 如果没有提供位置，使用节点索引作为位置
        if positions is None:
            positions = torch.arange(self.n_nodes, device=self.device)
        
        # 预计算所有节点的邻居聚合
        neighbor_aggregates = self._compute_neighbor_aggregates(H)

        # 向量化应用更新规则（使用边投影）
        if hasattr(self.update_rule, 'vectorized_forward_with_edges'):
            new_H = self.update_rule.vectorized_forward_with_edges(H, self.edge_index)
        else:
            new_H = self.update_rule.vectorized_forward(H, neighbor_aggregates)
        
        return new_H
    
    def _compute_neighbor_aggregates(self, H: torch.Tensor) -> torch.Tensor:
        """
        计算所有节点的邻居状态聚合（带边权重自适应和双向聚合）
        使用scatter_add进行高效聚合
        
        Args:
            H: 当前状态配置
            
        Returns:
            邻居聚合张量，形状为 [n_nodes, hidden_dim]
        """
        neighbor_aggregates = self._scatter_based_aggregation(H)
        
        return neighbor_aggregates
    
    
        
    
    def _scatter_based_aggregation(self, H: torch.Tensor) -> torch.Tensor:
        """
        使用scatter_add进行高效双向聚合（借鉴TreeFFN原型机）
        使用直接边存储，避免昂贵的链式前向星遍历
        
        Args:
            H: 当前状态配置
            
        Returns:
            邻居聚合张量，形状为 [n_nodes, hidden_dim]
        """
        # 初始化聚合结果
        agg = torch.zeros(self.n_nodes, self.hidden_dim, dtype=H.dtype, device=H.device)
        
        # 如果没有边权重参数，使用默认权重1.0
        if self.edge_weights is None:
            edge_weights = torch.ones(self.edge_count, device=H.device)
        else:
            # 应用sigmoid确保权重在0-1之间
            edge_weights = torch.sigmoid(self.edge_weights)
        
        # 直接使用edge_index进行高效聚合（类似TreeFFN）
        src_nodes = self.edge_index[0]  # 源节点索引
        tgt_nodes = self.edge_index[1]  # 目标节点索引
        
        # 计算每条边的消息（双向：源节点和目标节点的状态之和）
        edge_messages = (H[src_nodes] + H[tgt_nodes]) * edge_weights.unsqueeze(1)
        
        # 双向聚合：消息同时添加到源节点和目标节点
        agg.index_add_(0, src_nodes, edge_messages)
        agg.index_add_(0, tgt_nodes, edge_messages)
        
        return agg
    

    
    def check_convergence(self, H0: torch.Tensor, max_steps: int = 100, tol: float = 1e-6, 
                         positions: torch.Tensor = None) -> Dict:
        """
        检查收敛性（用于定理2.3验证）
        
        Args:
            H0: 初始状态
            max_steps: 最大迭代步数
            tol: 收敛容差
            positions: 位置索引，用于旋转位置编码
            
        Returns:
            收敛信息字典
        """
        current_H = H0.clone()
        prev_H = current_H.clone()
        
        # 如果没有提供位置，使用节点索引作为位置
        if positions is None:
            positions = torch.arange(self.n_nodes, device=self.device)
        
        convergence_info = {
            'converged': False,
            'steps': 0,
            'final_diff': float('inf'),
            'history': []
        }
        
        for _ in range(max_steps):
            current_H_state = self.single_step(current_H, positions)
            diff = torch.norm(current_H_state - prev_H).item()
            current_H = current_H_state
            
            convergence_info['history'].append(diff)
            
            if diff < tol:
                convergence_info['converged'] = True
                convergence_info['steps'] = _ + 1
                convergence_info['final_diff'] = diff
                break
                
            prev_H = current_H.clone()
            
            if _ == max_steps - 1:
                convergence_info['steps'] = max_steps
                convergence_info['final_diff'] = diff
        
        return convergence_info
    
    def encode_grid_to_tokens(self, grid: List[List[int]]) -> List[int]:
        """
        将单个网格编码为token序列（简化版本）

        Args:
            grid: 2D网格

        Returns:
            token序列: 直接展平，用分隔符分隔行
        """
        sequence = []
        for i, row in enumerate(grid):
            if i > 0:
                sequence.append(self.ROW_SEP)
            sequence.extend(row)

        return sequence

    def decode_tokens_to_grid(self, tokens: List[int]) -> List[List[int]]:
        """
        从token序列解码回网格（简化版本）

        Args:
            tokens: token序列

        Returns:
            2D网格
        """
        # 直接按分隔符分割序列
        rows = []
        current_row = []

        for token in tokens:
            if token == self.ROW_SEP:
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif 0 <= token <= 9:  # 颜色token
                current_row.append(token)

        if current_row:
            rows.append(current_row)

        # 确保至少有一行
        if not rows:
            return [[0]]

        return rows
    
    def encode_arc2_sequence(self, exam_inputs: List[List[List[int]]], exam_outputs: List[List[List[int]]],
                           test_input: List[List[int]], test_output: List[List[int]] = None,
                           max_length: int = 8192) -> List[int]:
        """
        编码ARC2数据格式为token序列（简化版本）

        Args:
            exam_inputs: 训练输入网格列表
            exam_outputs: 训练输出网格列表
            test_input: 测试输入网格
            test_output: 测试输出网格（如果为None则用-1填充）

        Returns:
            token序列: 直接拼接所有网格，用-1表示需要预测的部分
        """
        sequence = []

        # 编码训练对
        for exam_in, exam_out in zip(exam_inputs, exam_outputs):
            sequence.extend(self.encode_grid_to_tokens(exam_in))
            sequence.extend(self.encode_grid_to_tokens(exam_out))

        # 编码测试输入
        sequence.extend(self.encode_grid_to_tokens(test_input))

        # 编码测试输出（用PAD_TOKEN填充需要预测的部分）
        if test_output is not None:
            sequence.extend(self.encode_grid_to_tokens(test_output))
        else:
            # 预测模式：用PAD_TOKEN占位，但让模型动态预测输出长度
            # 使用max_length确保序列足够长，模型会学习预测正确的结束位置
            sequence.extend([self.PAD_TOKEN] * (max_length - len(sequence)))

        return sequence

    def parallel_generate(self, exam_inputs: List[List[List[int]]], exam_outputs: List[List[List[int]]],
                        test_input: List[List[int]], max_length: int = 8192, return_logits: bool = False):
        """
        并行生成ARC2序列 - 简化版本

        Args:
            exam_inputs: 训练输入网格列表
            exam_outputs: 训练输出网格列表
            test_input: 测试输入网格
            max_length: 最大序列长度
            return_logits: 是否返回logits

        Returns:
            predicted_test_output: 预测的测试输出网格
            如果return_logits=True，还返回完整的logits
        """
        if not self.has_parallel_generation:
            raise ValueError("Parallel generation components not initialized")

        # 编码完整序列（test_output用-1填充）
        input_tokens = self.encode_arc2_sequence(exam_inputs, exam_outputs, test_input, None, max_length)

        # 截断或填充到max_length（用PAD_TOKEN填充）
        if len(input_tokens) > max_length:
            input_tokens = input_tokens[:max_length]
        else:
            input_tokens = input_tokens + [self.PAD_TOKEN] * (max_length - len(input_tokens))

        input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)

        # 使用线性图结构而不是8邻居图，因为我们处理的是序列
        edge_index = self._create_linear_edges(len(input_tokens))
        n_nodes = len(input_tokens)

        # 获取token嵌入和位置编码
        if not hasattr(self, 'token_embedding_layer'):
            self.token_embedding_layer = nn.Embedding(self.vocab_size, self.hidden_dim, device=self.device)

        # 调试：检查输入tensor的范围
        invalid_tokens = input_tensor[(input_tensor < 0) | (input_tensor >= self.vocab_size)]
        if len(invalid_tokens) > 0:
            print(f"Warning: Found {len(invalid_tokens)} invalid tokens in input: {invalid_tokens.unique().tolist()}")
            # 将无效token替换为PAD_TOKEN
            input_tensor = torch.where((input_tensor < 0) | (input_tensor >= self.vocab_size),
                                     torch.tensor(self.PAD_TOKEN, device=self.device), input_tensor)

        token_emb = self.token_embedding_layer(input_tensor)
        positions = torch.arange(n_nodes, device=self.device)
        pos_emb = self.position_emb(positions)

        # 合并token和位置嵌入
        combined_emb = token_emb + pos_emb

        # 应用ASPP算子进行推理（使用8邻居图结构）
        self.set_graph(n_nodes, edge_index)
        processed_features = self.forward(combined_emb, K=5)

        # 预测所有位置的token
        output_logits = self.output_projection(processed_features)
        predicted_tokens = torch.argmax(output_logits, dim=-1)

        # 提取测试输出部分
        full_predicted_tokens = predicted_tokens.cpu().tolist()

        # 找到测试输出部分的起始位置（最后一个分隔符之后）
        try:
            # 使用行分隔符作为分隔标记
            last_sep_idx = len(input_tokens) - 1 - input_tokens[::-1].index(self.ROW_SEP)
            test_output_start = last_sep_idx + 1

            # 提取测试输出token并解码
            test_output_tokens = full_predicted_tokens[test_output_start:]

            # 动态检测序列结束：找到第一个连续的PAD_TOKEN序列作为结束
            # 模型应该学会预测ROW_SEP来标记网格行结束
            valid_tokens = []
            for token in test_output_tokens:
                if token == self.PAD_TOKEN:
                    # 遇到PAD_TOKEN，检查是否是序列结束
                    break
                valid_tokens.append(token)

            # 移除末尾的PAD_TOKEN（如果有）
            while valid_tokens and valid_tokens[-1] == self.PAD_TOKEN:
                valid_tokens.pop()

            predicted_test_output = self.decode_tokens_to_grid(valid_tokens)
        except ValueError:
            # 如果找不到分隔符，返回空网格
            predicted_test_output = [[0]]

        if return_logits:
            return predicted_test_output, output_logits
        else:
            return predicted_test_output
    
    def _create_8neighbor_edges(self, grid: List[List[int]]) -> torch.LongTensor:
        """
        创建8邻居网格连接边索引

        Args:
            grid: 2D网格

        Returns:
            边索引张量 [2, E]
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        edges = []

        # 将2D网格展平为1D序列
        def flatten_idx(i, j):
            return i * width + j

        # 方向邻居偏移
        directions = [
            (-1,-1),(-1, 0),(-1,1),
            (0, -1),    (0, 1),
            (1,-1),(1, 0),(1,1)
        ]

        for i in range(height):
            for j in range(width):
                current_idx = flatten_idx(i, j)

                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = flatten_idx(ni, nj)
                        edges.append([current_idx, neighbor_idx])

        if not edges:
            edges.append([0, 0])

        return torch.tensor(edges, dtype=torch.long, device=self.device).t()

    def _create_linear_edges(self, seq_len: int) -> torch.LongTensor:
        """创建线性图结构的边索引（双向连接）"""
        edges = []
        for i in range(seq_len - 1):
            edges.append([i, i + 1])  # 前向连接
            edges.append([i + 1, i])  # 后向连接

        if not edges:
            edges.append([0, 0])  # 单节点自连接

        return torch.tensor(edges, dtype=torch.long, device=self.device).t()
    
    def _decode_tokens_to_grid(self, tokens: List[int]) -> List[List[int]]:
        """从token序列解码回网格"""
        GRID_START = 10
        GRID_END = 11
        ROW_SEP = 12
        
        # 找到网格边界
        try:
            start_idx = tokens.index(GRID_START) + 1
            end_idx = tokens.index(GRID_END)
            grid_tokens = tokens[start_idx:end_idx]
        except ValueError:
            # 如果没有找到边界token，直接处理颜色token
            grid_tokens = [t for t in tokens if 0 <= t <= 9]
            if not grid_tokens:
                return [[0]]
        
        # 按行分隔符分割
        rows = []
        current_row = []
        
        for token in grid_tokens:
            if token == ROW_SEP:
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif 0 <= token <= 9:  # 颜色token
                current_row.append(token)
        
        if current_row:
            rows.append(current_row)
        
        # 确保至少有一行
        if not rows:
            return [[0]]
        
        return rows


# 测试代码
if __name__ == "__main__":
    # 测试ASPP算子
    device = torch.device('cpu')
    
    # 创建更新规则
    update_rule = FFNUpdate(64)
    aspp = ASPPOperator(update_rule, hidden_dim=64, device=device)
    
    # 创建简单的线性图边索引
    n_nodes = 5
    edge_index = torch.tensor([[0, 1, 2, 3],  # 源节点
                              [1, 2, 3, 4]],  # 目标节点
                             dtype=torch.long, device=device)
    
    # 设置图结构
    aspp.set_graph(n_nodes, edge_index)
    
    # 初始化状态
    H0 = torch.randn(5, 64, device=device)
    
    # 单步更新
    H1 = aspp.single_step(H0)
    print(f"Single step update: {H0.shape} -> {H1.shape}")
    
    # K步演化
    H_final = aspp.forward(H0, K=10)  # 使用初始K值
    print(f"10-step evolution successful: {H_final.shape}")
    
    # 收敛性测试
    conv_info = aspp.check_convergence(H0, max_steps=15)  # 使用最大K步
    print(f"Convergence info: {conv_info}")
    
    # 测试并行生成（使用新接口）
    print("\nTesting parallel generation:")
    exam_inputs = [[[1, 2], [3, 4]]]
    exam_outputs = [[[2, 3], [4, 5]]]
    test_input = [[1, 2], [3, 4]]

    predicted_grid = aspp.parallel_generate(exam_inputs, exam_outputs, test_input)
    print(f"Test input: {test_input}")
    print(f"Predicted output: {predicted_grid}")
    print("Parallel generation successful!")

    # 重新设置主算子的图结构（因为parallel_generate会修改它）
    aspp.set_graph(n_nodes, edge_index)

    # 测试多层算子实例功能
    print("\nTesting multi-instance operator functionality:")

    # 创建子算子实例
    child_op1 = aspp.create_child_operator("child1", hidden_dim=64)
    child_op2 = aspp.create_child_operator("child2", hidden_dim=64)

    # 设置相同的图结构
    child_op1.set_graph(n_nodes, edge_index)
    child_op2.set_graph(n_nodes, edge_index)

    # 测试子算子功能
    H_child1 = child_op1.forward(H0, K=2)
    H_child2 = child_op2.forward(H0, K=2)
    print(f"Child operator 1 output shape: {H_child1.shape}")
    print(f"Child operator 2 output shape: {H_child2.shape}")

    # 测试算子栈
    aspp.add_to_stack(child_op1)
    aspp.add_to_stack(child_op2)

    H_stack = aspp.forward(H0, K=2, use_stack=True)
    print(f"Operator stack output shape: {H_stack.shape}")
    print("Multi-instance operator functionality successful!")

    # 测试ARC2数据格式处理
    print("\nTesting ARC2 data format processing:")

    # 创建训练示例
    exam_inputs = [
        [[1, 2], [3, 4]],  # 训练输入1
        [[0, 5], [6, 7]]   # 训练输入2
    ]
    exam_outputs = [
        [[2, 3], [4, 5]],  # 训练输出1
        [[1, 6], [7, 8]]   # 训练输出2
    ]

    # 测试输入
    test_input = [[8, 9], [0, 1]]

    # 测试编码
    encoded_seq = aspp.encode_arc2_sequence(exam_inputs, exam_outputs, test_input)
    print(f"Encoded sequence length: {len(encoded_seq)}")
    print(f"Encoded sequence: {encoded_seq[:20]}...")  # 显示前20个token

    # 测试解码
    decoded_test = aspp.decode_tokens_to_grid(encoded_seq)
    print(f"Decoded test grid: {decoded_test}")

    # 测试8邻居边创建
    edge_index = aspp._create_linear_edges(len(test_input))
    print(f"8-neighbor edges shape: {edge_index.shape}")

    print("ARC2 data format processing successful!")
