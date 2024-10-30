import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
print("importing v2m")
from v2m import V2MTransformer
print("importing tqdm")
from tqdm import tqdm
from dataloader import load_data, V2MDataset, collate_fn
import json
import gc
import matplotlib.pyplot as plt
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# 修改训练函数
def train(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio=0.5, accumulation_steps=16):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (fixed_tokens, two_numbers, variable_tokens, targets_with_eos, input_mask, target_mask) in enumerate(tqdm(train_loader)):
        # 检查输入数据
        # if i >= 200:
        #     torch.autograd.set_detect_anomaly(True)
        fixed_tokens = fixed_tokens.to(device)
        two_numbers = two_numbers.to(device)
        variable_tokens = variable_tokens.to(device)
        targets_with_eos = targets_with_eos.to(device)
        input_mask = input_mask.to(device)
        target_mask = target_mask.to(device)
        #找到target_mask中为1的元素数量
        outputs = model(fixed_tokens, two_numbers, variable_tokens, input_mask, targets_with_eos, target_mask, teacher_forcing_ratio)
        # 检查输出
        if i % 40 == 0:
            print("Sample output:", outputs[0, :5, :5]) # 打印第一个样本的前5个token前5个值
            print(outputs[target_mask][-5:,:])
            print("Sample target:", targets_with_eos[0, :5, :5])
        
        loss = calculate_loss(outputs, targets_with_eos, target_mask, criterion)
        loss = loss / accumulation_steps
        loss.backward()
        print("grade norm:", torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None]), 2))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            print(outputs)
            
            # 确保输出和目标具有相同的形状
            outputs = outputs[:, :targets_with_eos.size(1), :]
            
            loss = calculate_loss(outputs, targets_with_eos, target_mask, criterion)
            total_loss += loss.item()
            print("Validation Loss:", loss.item())

    return total_loss / len(val_loader)

def calculate_loss(outputs, targets, target_mask, criterion):
    # 只选择被掩码的部分
    masked_outputs = outputs[target_mask]
    masked_targets = targets[target_mask]
    # 计算损失
    if masked_outputs.numel() > 0:
        loss = criterion(masked_outputs, masked_targets)
        return loss
    else:
        return torch.tensor(0.0, device=outputs.device)

def adjust_teacher_forcing_ratio(epoch, total_epochs):
    return max(0, 0.2 - 0.4 * (epoch / total_epochs))
# 主函数
def main():
    # 设置设备
    print("Main function started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # 加载数据
    print("Loading data...")
    with open('config.json', 'r') as f:
        config = json.load(f)
    full_dataloader, max_target_length = load_data(config)
    full_dataset = full_dataloader.dataset
    print("Data loaded.")
    print(f"Max target length: {max_target_length}")

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
    model = V2MTransformer(max_target_length=max_target_length+1).to(device)
    model_path = "/home/yihan/v2m/best_model_mid_3_1.pth"  # 替换为您的模型路径
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # model.apply(init_weights)
    print("Model initialized.")
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 或者您选择的其他损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # 学习率调整
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 训练循环
    num_epochs = 20
    teacher_forcing_ratio = 1
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
        # 将训练损失和验证损失写入文件
        import os
        
        # 如果文件不存在则创建
        if not os.path.exists('training_losses.txt'):
            open('training_losses.txt', 'w').close()
            
        with open('training_losses.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n')
        best_model = model.state_dict()
        torch.save(best_model, f"best_model_mid_3_1_{epoch}.pth")

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
    print("Clearing memory...")
    clear_memory()
    print("Starting main function...")
    main()
