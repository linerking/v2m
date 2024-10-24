import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from v2m import V2MTransformer
from tqdm import tqdm
from dataloader import load_data, V2MDataset, collate_fn
import json
import gc
import matplotlib.pyplot as plt

# 修改训练函数
def train(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio=0.5, accumulation_steps=8):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (fixed_tokens, two_numbers, variable_tokens, targets_with_eos, input_mask, target_mask) in enumerate(tqdm(train_loader)):
        fixed_tokens = fixed_tokens.to(device)
        two_numbers = two_numbers.to(device)
        variable_tokens = variable_tokens.to(device)
        targets_with_eos = targets_with_eos.to(device)
        input_mask = input_mask.to(device)
        target_mask = target_mask.to(device)
        
        outputs = model(fixed_tokens, two_numbers, variable_tokens, input_mask, targets_with_eos, teacher_forcing_ratio)
        
        loss = calculate_loss(outputs, targets_with_eos, target_mask, criterion)
        loss = loss / accumulation_steps
        loss.backward()
        print("Loss:", loss.item())
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(train_loader)

# 修改验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for fixed_tokens, two_numbers, variable_tokens, targets_with_eos, input_mask, target_mask in tqdm(val_loader):
            fixed_tokens = fixed_tokens.to(device)
            two_numbers = two_numbers.to(device)
            variable_tokens = variable_tokens.to(device)
            targets_with_eos = targets_with_eos.to(device)
            input_mask = input_mask.to(device)
            target_mask = target_mask.to(device)

            outputs = model(fixed_tokens, two_numbers, variable_tokens, input_mask)
            
            
            # 确保输出和目标具有相同的形状
            outputs = outputs[:, :targets_with_eos.size(1), :]
            
            loss = criterion(outputs, targets_with_eos)
            loss = loss.view(targets_with_eos.shape)
            loss = (loss * target_mask.unsqueeze(-1)).sum() / (target_mask.sum() * targets_with_eos.size(-1))
            total_loss += loss.item()
            print("Validation Loss:", loss.item())

    return total_loss / len(val_loader)

def calculate_loss(outputs, targets, target_mask, criterion, length_penalty_factor=0.5):
    # 获取输出和目标的序列长度
    output_length = outputs.size(1)
    target_length = targets.size(1)
    
    # 计算共同的长度
    min_length = min(output_length, target_length)
    max_length = max(output_length, target_length)
    
    # 计算共同部分的损失
    common_loss = criterion(outputs[:, :min_length, :], targets[:, :min_length, :])
    common_loss = (common_loss * target_mask[:, :min_length].unsqueeze(-1)).sum() / (target_mask[:, :min_length].sum() * targets.size(-1))
    
    # 计算长度差异
    length_diff = abs(output_length - target_length)
    
    # 应用长度惩罚
    length_penalty = length_diff * length_penalty_factor
    
    # 总损失
    total_loss = common_loss + length_penalty
    
    return total_loss
def adjust_teacher_forcing_ratio(epoch, total_epochs):
    return max(0.5, 0.9 - 0.4 * (epoch / total_epochs))
# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # 加载数据
    print("Loading data...")
    with open('config.json', 'r') as f:
        config = json.load(f)
    full_dataloader = load_data(config)
    full_dataset = full_dataloader.dataset
    print("Data loaded.")

    # 计算训练集和验证集的大小
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    print(f"Total dataset size: {dataset_size}")
    print(f"Train size: {train_size}, Val size: {val_size}")

    # 创建训练集和验证集的索引
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 创建 Subset 对象
    print("full_dataset created")
    train_dataset = Subset(full_dataset, train_indices)
    print("train_dataset created")
    val_dataset = Subset(full_dataset, val_indices)
    print("val_dataset created")
    # 定义批次大小
    batch_size = 4  # 或者 8，取决于您的 GPU 内存
    
    # 创建新的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    print("train_loader created")
    print("val_loader created")
    # 初始化模型
    model = V2MTransformer().to(device)
    print("Model initialized.")
    # 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学习率调整
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 训练循环
    num_epochs = 10
    teacher_forcing_ratio = 0.9
    best_val_loss = float('inf')
    patience = 2
    no_improve = 0
    best_model = None

    # 在 main 函数开始处添加这些列表
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        teacher_forcing_ratio = adjust_teacher_forcing_ratio(epoch, num_epochs)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio)
        val_loss = validate(model, val_loader, criterion, device)
        
        # 收集损失数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 在每个epoch结束后进行垃圾回收
        
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_model = model.state_dict()
            torch.save(best_model, "best_model.pth")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # 训练结束后，绘制损失图表
    plot_losses(train_losses, val_losses)

    # 训练结束后，加载最佳模型
    model.load_state_dict(best_model)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 在关键点调用此函数

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()

if __name__ == "__main__":
    clear_memory()
    main()
