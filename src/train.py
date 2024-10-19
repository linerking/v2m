import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from v2m import V2MTransformer
import numpy as np
from tqdm import tqdm

# 假设的数据集类
class V2MDataset(Dataset):
    def __init__(self, fixed_tokens, two_numbers, variable_tokens, targets):
        self.fixed_tokens = fixed_tokens
        self.two_numbers = two_numbers
        self.variable_tokens = variable_tokens
        self.targets = targets

    def __len__(self):
        return len(self.fixed_tokens)

    def __getitem__(self, idx):
        return (self.fixed_tokens[idx], self.two_numbers[idx], 
                self.variable_tokens[idx], self.targets[idx])

# 数据加载函数
def load_data():
    # 这里应该是实际的数据加载逻辑
    # 以下只是示例
    fixed_tokens = np.random.randn(1000, 10, 768)
    two_numbers = np.random.randn(1000, 2)
    variable_tokens = np.random.randn(1000, 20, 768)
    targets = np.random.randn(1000, 30, 1536)  # 假设最大输出长度为30
    return V2MDataset(fixed_tokens, two_numbers, variable_tokens, targets)

def add_eos_to_targets(targets, eos_embedding, pad_value=0):
    batch_size, max_seq_len, dim = targets.shape
    eos = eos_embedding.expand(batch_size, 1, dim)
    
    # 创建一个新的张量来存储带有 EOS 的目标
    new_targets = torch.full((batch_size, max_seq_len + 1, dim), pad_value, device=targets.device)
    
    # 找到每个序列的实际长度（非填充的部分）
    seq_lengths = torch.sum(targets.abs().sum(dim=-1) != 0, dim=1)
    
    # 复制原始序列的非填充部分
    for i in range(batch_size):
        new_targets[i, :seq_lengths[i]] = targets[i, :seq_lengths[i]]
        new_targets[i, seq_lengths[i]] = eos[i]
    
    return new_targets

def create_mask(fixed_tokens, two_numbers, variable_tokens, targets):
    batch_size = fixed_tokens.size(0)
    fixed_length = fixed_tokens.size(1)
    variable_length = variable_tokens.size(1)
    total_input_length = fixed_length + 1 + variable_length  # 1 是为 two_numbers 预留的位置
    target_length = targets.size(1)

    # 创建输入掩码
    input_mask = torch.ones(batch_size, total_input_length, dtype=torch.bool, device=fixed_tokens.device)

    # 创建目标掩码
    target_mask = torch.zeros(batch_size, target_length + 1, dtype=torch.bool, device=targets.device)

    # 找到每个序列的实际长度（非填充的部分）
    input_lengths = fixed_length + 1 + torch.sum(variable_tokens.abs().sum(dim=-1) != 0, dim=1)
    target_lengths = torch.sum(targets.abs().sum(dim=-1) != 0, dim=1)

    # 更新掩码以反映实际长度
    for i in range(batch_size):
        input_mask[i, :input_lengths[i]] = True
        target_mask[i, :target_lengths[i] + 1] = True  # +1 for EOS

    return input_mask, target_mask

# 修改训练函数
def train(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    for fixed_tokens, two_numbers, variable_tokens, targets in tqdm(train_loader):
        fixed_tokens = fixed_tokens.to(device)
        two_numbers = two_numbers.to(device)
        variable_tokens = variable_tokens.to(device)
        targets = targets.to(device)

        # 添加 EOS 到目标序列
        targets_with_eos = add_eos_to_targets(targets, model.eos_embedding, pad_value=0)

        # 创建输入掩码和目标掩码
        input_mask, target_mask = create_mask(fixed_tokens, two_numbers, variable_tokens, targets)

        optimizer.zero_grad()
        outputs = model(fixed_tokens, two_numbers, variable_tokens, input_mask, targets_with_eos, teacher_forcing_ratio)
        
        # 使用目标掩码来计算损失
        loss = criterion(outputs, targets_with_eos)
        loss = (loss * target_mask.unsqueeze(-1)).sum() / target_mask.sum()  # 平均损失只计算非填充位置

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

# 修改验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for fixed_tokens, two_numbers, variable_tokens, targets in tqdm(val_loader):
            fixed_tokens = fixed_tokens.to(device)
            two_numbers = two_numbers.to(device)
            variable_tokens = variable_tokens.to(device)
            targets = targets.to(device)

            # 添加 EOS 到目标序列
            targets_with_eos = add_eos_to_targets(targets, model.eos_embedding, pad_value=0)

            # 创建输入掩码和目标掩码
            input_mask, target_mask = create_mask(fixed_tokens, two_numbers, variable_tokens, targets)

            outputs = model(fixed_tokens, two_numbers, variable_tokens, input_mask)
            
            # 使用目标掩码来计算损失
            loss = criterion(outputs, targets_with_eos)
            loss = (loss * target_mask.unsqueeze(-1)).sum() / target_mask.sum()  # 平均损失只计算非填充位置

            total_loss += loss.item()
    return total_loss / len(val_loader)

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataset = load_data()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # 初始化模型
    model = V2MTransformer().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 10
    teacher_forcing_ratio = 0.5
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 可选：逐步减少 teacher forcing 比率
        teacher_forcing_ratio *= 0.9

        # 保存模型
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
