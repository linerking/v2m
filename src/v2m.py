import torch
import torch.nn as nn
import math
import random

class V2MTransformer(nn.Module):
    def __init__(self, input_dim=768, output_dim=1536, nhead=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(V2MTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 512  # 新增的中间维度
        
        # 为三种输入类型创建单独的MLP
        self.fixed_tokens_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.two_numbers_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.variable_tokens_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.positional_encoding = PositionalEncoding(output_dim, dropout=dropout)
        self.eos_embedding = torch.ones(1, 1, output_dim)
        
    def forward(self, fixed_tokens, two_numbers, variable_tokens, input_mask, target=None, teacher_forcing_ratio=0.5):
        # 对每种输入类型进行投影
        fixed_embed = self.fixed_tokens_mlp(fixed_tokens)
        number_embed = self.two_numbers_mlp(two_numbers).unsqueeze(1)  # 添加一个维度
        variable_embed = self.variable_tokens_mlp(variable_tokens)
        
        # 拼接所有输入
        combined_input = torch.cat([fixed_embed, number_embed, variable_embed], dim=1)
        combined_input = self.positional_encoding(combined_input)
        
        memory_key_padding_mask = ~input_mask
        
        memory = combined_input.permute(1, 0, 2)
        
        if target is not None:
            return self.train_generate(memory, memory_key_padding_mask, target, teacher_forcing_ratio)
        else:
            return self.generate(memory, memory_key_padding_mask)
    
    def train_generate(self, memory, memory_key_padding_mask, target, teacher_forcing_ratio):
        batch_size, max_len, _ = target.size()
        device = memory.device
        
        decoder_input = torch.zeros(1, batch_size, self.output_dim, device=device)
        outputs = torch.zeros(batch_size, max_len, self.output_dim, device=device)
        
        for t in range(max_len):
            decoder_input = self.positional_encoding(decoder_input)
            output = self.decoder(decoder_input, memory, memory_key_padding_mask=memory_key_padding_mask)
            
            outputs[:, t, :] = output.squeeze(0)
            
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target[:, t, :].unsqueeze(0) if teacher_force else output
        
        return outputs
    
    def generate(self, memory, memory_key_padding_mask, max_gen_len=256):
        batch_size = memory.size(1)
        device = memory.device
        
        decoder_input = torch.zeros(1, batch_size, self.output_dim, device=device)
        outputs = []
        
        for _ in range(max_gen_len):
            decoder_input = self.positional_encoding(decoder_input)
            output = self.decoder(decoder_input, memory, memory_key_padding_mask=memory_key_padding_mask)
            
            if torch.cosine_similarity(output.squeeze(0), self.eos_embedding.to(device).squeeze(0), dim=-1).mean() > 0.9:
                break
            
            outputs.append(output.squeeze(0))
            decoder_input = output
        
        outputs = torch.stack(outputs, dim=1)
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        # 创建一个长度为max_len，每个元素包含d_model个值的tensor
        pe = torch.zeros(max_len, d_model)
        # 创建一个表示位置的tensor
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建一个用于计算位置编码的除数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度并转置
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将位置编码注册为一个缓冲区，这样它就不会被认为是模型参数
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 将位置编码添加到输入中
        return x + self.pe[:x.size(0), :] + self.dropout(x)

# 使用示
input_dim = 768
output_dim = 1536
batch_size = 32
fixed_length = 10
variable_length = 20

model = V2MTransformer(input_dim=input_dim, output_dim=output_dim)

# 创建模拟输入数据
fixed_tokens = torch.randn(batch_size, fixed_length, input_dim)
two_numbers = torch.randn(batch_size, 2)  # 两个数字
variable_tokens = torch.randn(batch_size, variable_length, input_dim)

# 创建输入掩码
total_length = fixed_length + 1 + variable_length  # 1 是为 two_numbers 预留的位置
input_mask = torch.ones(batch_size, total_length, dtype=torch.bool)
# 如果需要模拟不同长度的输入，可以随机设置一些位置为 False
# 例如：input_mask[:, -5:] = False  # 将最后5个位置设为填充

# 运行模型
output = model(fixed_tokens, two_numbers, variable_tokens, input_mask)
print(output.shape)
