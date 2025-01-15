import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from base_model import SequenceModel

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

# bert的多头注意力模块
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        # self.norm = LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.o_linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_state, attention_mask=None):
        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        dim = int(self.d_model/self.n_heads)
        batch_size = hidden_state.size(0)  # 获取批量大小
        # 计算 Q、K、V，线性变换
        query = self.Wq(hidden_state)  # (batch_size, seq_len, hidden_size)
        # print (query)

        key = self.Wk(hidden_state)    # (batch_size, seq_len, hidden_size)
        value = self.Wv(hidden_state)  # (batch_size, seq_len, hidden_size)
        # print (query)
        # 分割多头，将每个头的维度拆分出来
        query = self.split_head(query, dim)  # (batch_size, num_heads, seq_len, head_dim)
        
        key = self.split_head(key,dim)      # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_head(value,dim)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数，使用缩放点积注意力机制
        # attention_scores 形状: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        
        # 添加注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷
        if attention_mask is not None:
            attention_scores += attention_mask * -1e9
        
        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        # drop the attention_probs
        attention_probs = self.dropout(attention_probs)
        # 计算注意力输出，通过注意力概率加权值
        output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 对多头注意力输出进行拼接
        # output.transpose(1, 2) 将 num_heads 和 seq_len 维度转置
        # 将形状调整为 (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(batch_size, -1, dim * self.n_heads)
        
        # 通过线性层将拼接后的输出变换为所需的输出维度
        output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
        return output
    def split_head(self, x, dim):
        batch_size = x.size(0)  # 获取批量大小
        # x 形状: (batch_size, seq_len, hidden_size)
        # 将 hidden_size 分割为 num_heads 和 head_dim
        return x.reshape(batch_size, -1, self.n_heads, dim).transpose(1, 2)
        # 返回形状: (batch_size, num_heads, seq_len, head_dim)



# 前馈网络模块
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        # position:(max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # self.register_buffer 是一个用于向模型（通常是nn.Module的子类）注册一个不需要梯度的张量（tensor）的方法。
        # 注册的张量将会保存在state_dict中，并在模型保存和加载时被持久化。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]：天数、T历史窗口、F特征数
        # self.pe[: x.size(0), :]=> [T, 1, F]
        # print(self.pe.shape)
        # print(x.shape)
        return x + self.pe[:x.shape[1], :]
    
class AddNormLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNormLayer, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = self.dropout(input)
        return self.norm(input)

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # 753,8,256
        h = self.trans(z) # [N, T, D]
        # 753,256,1
        query = h[:, -1, :].unsqueeze(-1)
        # 753,1,8
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output
    
class Transformer(nn.Module):
    def __init__(self, d_ff=158, d_model=512, dropout=0.1, n_layers=6, device=None):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.device = device
        self.feature_layer = nn.Linear(d_ff, d_model)
        self.posi = PositionalEncoding(d_model, max_len=100)
        self.attn = MultiHeadAttention(n_heads=8, d_model=d_model, dropout=0.1)
        self.addNorm = AddNormLayer(d_model, dropout=0.1)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=0.1)
        self.temporal_att = TemporalAttention(d_model=d_model)
        self.output = nn.Linear(d_model,1)

    def forward(self, src):
        feature = self.feature_layer(src.float())
        posi_out=self.posi(feature)
        attn_out=self.attn(posi_out)
        addNorm_out1 = attn_out + self.addNorm(attn_out)
        ffn_out = self.ffn(addNorm_out1)
        addNorm_out2 = ffn_out + self.addNorm(ffn_out)
        temporal_att_out = self.temporal_att(addNorm_out2)
        output = self.output(temporal_att_out)  # [batch_size, seq_len, 1]
        return output

class TransformerModel(SequenceModel):
    def __init__(self, d_feat, d_model, n_heads, lr, **kwargs):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.n_heads = n_heads
        self.lr = lr
        self.init_model()

    def init_model(self):
        self.model = Transformer(d_model=self.d_model, d_ff=self.d_feat, n_heads=self.n_heads, lr=self.lr, device=self.device)

def train_transformer(dataloader, epoch, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # if epoch == 0:
    #     model.load_state_dict(torch.load(f'params_log/transformer_model_epoch50.pkl', map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # for name, parameters in model.named_parameters(): 
    #     print(name, ':', parameters)
    losses = []
    for batch_idx,data in enumerate(dataloader):
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1].to(device)
        pred = model(feature.float())
        # print(pred)
        loss = torch.mean((pred - label) ** 2)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
        # print(f'batch_idx:{batch_idx}, loss: {loss.item()},feature: {feature.shape}')
    if(epoch == 50):
        torch.save(model.state_dict(), f'params_log/transformer_model_epoch{epoch}.pkl')
        
    return float(np.mean(losses))

def test_transformer(dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Transformer()
    model.to(device)
    losses = []
    scores = []
    for batch_idx, data in enumerate(dataloader):
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1].to(device)
        with torch.no_grad():
            pred = model(feature.float())
            loss = torch.mean((pred - label) ** 2)
            losses.append(loss.item())
            mask = ~torch.isnan(label)
            score = torch.mean((pred[mask] - label[mask]) ** 2)
            scores.append(score.item())
    return float(np.mean(losses))