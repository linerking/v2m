import torch
import torch.nn as nn
import math
import random

class V2MTransformer(nn.Module):
    def __init__(self, input_dim=768, output_dim=1536, nhead=4, num_decoder_layers=6, dim_feedforward=4096, dropout=0.1, max_target_length=136):
        super(V2MTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 2048  # 新增的中间维度
        
        # 为三种输入类型创建单独的MLP
        
        self.two_numbers_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.positional_encoding = PositionalEncoding(input_dim, dropout=dropout)
        
        self.max_target_length = max_target_length
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, fixed_tokens, two_numbers, variable_tokens, input_mask, target=None, target_mask=None, teacher_forcing_ratio=0.5):
        # 对每种输入类型进行投影
        
        number_embed = self.two_numbers_mlp(two_numbers).unsqueeze(1)
        
        
        # 拼接所有输入
        #combined_input = torch.cat([fixed_tokens, number_embed, variable_tokens], dim=1)
        combined_input = torch.cat([fixed_tokens, number_embed], dim=1)
        combined_input = self.positional_encoding(combined_input)
        
        memory_key_padding_mask = ~input_mask
        
        memory = combined_input.permute(1, 0, 2)
        
        if target is not None:
            outputs = self.train_generate(memory, memory_key_padding_mask, target, target_mask, teacher_forcing_ratio)
        else:
            outputs = self.generate(memory, memory_key_padding_mask)
        return outputs
    
    def train_generate(self, memory, memory_key_padding_mask, target, target_mask, teacher_forcing_ratio):
        batch_size, max_target_length, _ = target.size()
        device = memory.device
        decoder_input = torch.zeros(1, batch_size, self.input_dim, device=device)
        outputs = []
        
        for t in range(max_target_length):
            decoder_input = self.positional_encoding(decoder_input)
            output = self.decoder(decoder_input, memory, memory_key_padding_mask=memory_key_padding_mask)
            
            last_output = output[-1]  # 已经是 [batch_size, input_dim]
        
            projected_output = self.output_projection(last_output)
            outputs.append(projected_output)
            if t < max_target_length - 1:
                teacher_force = random.random() < teacher_forcing_ratio
                if teacher_force:
                    next_input = torch.where(target_mask[:, t].bool().unsqueeze(1), 
                                             self.target_encoder(target[:, t, :]), 
                                             self.target_encoder(projected_output))
                else:
                    next_input = self.target_encoder(projected_output)
                decoder_input = torch.cat([decoder_input, next_input.unsqueeze(0)], dim=0)
            
            decoder_input = decoder_input[-self.max_target_length:]
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def generate(self, memory, memory_key_padding_mask):
        batch_size = memory.size(1)
        device = memory.device
        
        decoder_input = torch.zeros(1, batch_size, self.input_dim, device=device)
        outputs = []
        
        for _ in range(self.max_target_length):
            decoder_input = self.positional_encoding(decoder_input)
            output = self.decoder(decoder_input, memory, memory_key_padding_mask=memory_key_padding_mask)
            
            last_output = output[-1]  # 已经是 [batch_size, input_dim]
            
            projected_output = self.output_projection(last_output)
            outputs.append(projected_output)
            # if torch.allclose(projected_output, torch.zeros_like(projected_output), atol=1e-2):
            #     break
            next_input = self.target_encoder(projected_output)
            decoder_input = torch.cat([decoder_input, next_input.unsqueeze(0)], dim=0)
            
            decoder_input = decoder_input[-self.max_target_length:]
        
        outputs = torch.stack(outputs, dim=1)
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# # 使用示
# input_dim = 768
# output_dim = 1536
# batch_size = 32
# fixed_length = 10
# variable_length = 20

# model = V2MTransformer(input_dim=input_dim, output_dim=output_dim)

# # 创建模拟输入数据
# fixed_tokens = torch.randn(batch_size, fixed_length, input_dim)
# two_numbers = torch.randn(batch_size, 2)  # 两个数字
# variable_tokens = torch.randn(batch_size, variable_length, input_dim)

# # 创建输入掩码
# total_length = fixed_length + 1 + variable_length  # 1 是为 two_numbers 预留的位置
# input_mask = torch.ones(batch_size, total_length, dtype=torch.bool)
# # 如果需要模拟不长度的输入，可以随机设置一些位置为 False
# # 例如：input_mask[:, -5:] = False  # 将最后5个位置设为填充

# # 运行模型
# output = model(fixed_tokens, two_numbers, variable_tokens, input_mask)
# print(output.shape)
