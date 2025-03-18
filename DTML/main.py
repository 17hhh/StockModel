import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """论文中的位置编码模块（新增）"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DataAxisAttention(nn.Module):
    """论文核心创新点：数据轴注意力机制"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.stock_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.time_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, stock_feat, time_feat):
        # Stock-axis attention
        stock_out, _ = self.stock_attn(stock_feat, stock_feat, stock_feat)
        
        # Time-axis attention
        time_out, _ = self.time_attn(time_feat, time_feat, time_feat)
        
        return stock_out + time_out

class ContextEncoder(nn.Module):
    """上下文编码器（包含论文中的LSTM+Attention）"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # LSTM编码
        out, (h_n, c_n) = self.lstm(x)
        
        # 注意力机制
        att_weights = F.softmax(self.attention(out), dim=1)
        context = torch.sum(att_weights * out, dim=1)
        
        return context

class MultiLevelContextFusion(nn.Module):
    """多层次上下文融合模块（论文核心）"""
    def __init__(self, stock_dim, macro_dim, hidden_dim):
        super().__init__()
        # 个股上下文编码
        self.stock_encoder = ContextEncoder(stock_dim, hidden_dim)
        
        # 宏观上下文编码
        self.macro_encoder = ContextEncoder(macro_dim, hidden_dim)
        
        # 动态融合权重
        self.fusion_weight = nn.Parameter(torch.randn(1))
        
        # 上下文归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, stock_input, macro_input):
        stock_context = self.stock_encoder(stock_input)
        macro_context = self.macro_encoder(macro_input)
        
        # 多级融合
        fused = self.norm(stock_context + self.fusion_weight * macro_context)
        return fused

class DATP(nn.Module):
    """Data-Axis Transformer Predictor（论文完整模型）"""
    def __init__(self, 
                 stock_feat_dim=10,   # 个股特征维度
                 macro_feat_dim=5,    # 宏观特征维度
                 num_stocks=500,      # 股票数量
                 window_size=30,      # 时间窗口
                 hidden_dim=64,       # 隐藏维度
                 num_heads=8,         # 注意力头数
                 dropout=0.1):        
        super().__init__()
        
        # 1. 特征嵌入层
        self.stock_embed = nn.Linear(stock_feat_dim, hidden_dim)
        self.macro_embed = nn.Linear(macro_feat_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 2. 多层次上下文融合
        self.context_fusion = MultiLevelContextFusion(
            stock_dim=hidden_dim,
            macro_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # 3. 数据轴注意力
        self.data_axis_attn = DataAxisAttention(hidden_dim, num_heads)
        
        # 4. 非线性增强模块
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        
        # 5. 预测层
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)  # 二分类输出
        )
        
    def forward(self, stock_input, macro_input):
        """
        输入维度：
        stock_input: [batch_size, num_stocks, window_size, stock_feat_dim]
        macro_input: [batch_size, window_size, macro_feat_dim]
        """
        batch_size = stock_input.size(0)
        
        # 1. 特征嵌入
        stock_embedded = self.stock_embed(stock_input)  # [B, N, W, H]
        macro_embedded = self.macro_embed(macro_input)  # [B, W, H]
        
        # 添加位置编码
        # stock_embedded = self.pos_encoder(stock_embedded)
        
        # 2. 时间轴处理
        stock_processed = stock_embedded.mean(dim=2)    # [B, N, H]
        macro_processed = macro_embedded.mean(dim=1)    # [B, H]
        
        # 3. 多层次上下文融合
        context = self.context_fusion(stock_processed, macro_processed)  # [B, H]
        
        # 4. 数据轴注意力
        stock_feat = stock_processed.unsqueeze(1)       # [B, 1, N, H]
        time_feat = stock_embedded                      # [B, N, W, H]
        attn_out = self.data_axis_attn(stock_feat, time_feat)  # [B, N, H]
        
        # 5. 残差连接+MLP
        enhanced = self.mlp(attn_out + context.unsqueeze(1))
        
        # 6. 预测输出
        pred = self.pred_head(enhanced)  # [B, N, 2]
        
        return F.softmax(pred, dim=-1)

# 示例使用
if __name__ == "__main__":
    # 输入参数
    batch_size = 32
    num_stocks = 500
    window_size = 30
    stock_feat_dim = 10
    macro_feat_dim = 5
    
    # 创建模型
    model = DATP(
        stock_feat_dim=stock_feat_dim,
        macro_feat_dim=macro_feat_dim,
        num_stocks=num_stocks
    )
    
    # 生成测试数据
    stock_input = torch.randn(batch_size, num_stocks, window_size, stock_feat_dim)
    macro_input = torch.randn(batch_size, window_size, macro_feat_dim)
    
    # 前向传播
    output = model(stock_input, macro_input)
    
    print(f"输入维度：Stock: {stock_input.shape}, Macro: {macro_input.shape}")
    print(f"输出维度：{output.shape}")  # 预期输出：[32, 500, 2]