import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DigitMoE(nn.Module):
    def __init__(self, input_size=784, num_experts=10):
        super().__init__()
        self.num_experts = num_experts
        
        # 每个专家专门负责一个数字
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平图像
        
        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.squeeze(-1)  # [batch_size, 10]
        
        # 获取门控权重
        gate_scores = F.softmax(self.gate(x), dim=-1)
        
        # 最终输出
        output = gate_scores * expert_outputs
        
        return output, gate_scores, expert_outputs

def train_digit_moe():
    # 加载MNIST数据集   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitMoE().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)  # 获取当前批次的大小
            
            optimizer.zero_grad()
            
            output, gate_scores, expert_outputs = model(data)
            
            # 1. 分类损失
            classification_loss = F.cross_entropy(output, target)
            
            # 2. 专家对应损失：确保每个专家主要响应其对应的数字
            # 创建目标矩阵：每个样本只有对应数字的专家应该有高响应
            expert_target = torch.zeros_like(expert_outputs)
            for i in range(batch_size):
                expert_target[i, target[i]] = 1
            
            expert_loss = F.mse_loss(F.sigmoid(expert_outputs), expert_target)
            
            # 3. 门控对应损失：确保门控网络选择正确的专家
            gate_target = torch.zeros_like(gate_scores)
            for i in range(batch_size):
                gate_target[i, target[i]] = 1
            
            gate_loss = F.mse_loss(gate_scores, gate_target)
            
            # 总损失
            total_loss = classification_loss + expert_loss + gate_loss
            
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}')
                print(f'Classification Loss: {classification_loss.item():.4f}')
                print(f'Expert Loss: {expert_loss.item():.4f}')
                print(f'Gate Loss: {gate_loss.item():.4f}')
                
                # # 打印每个专家对不同数字的平均响应
                # with torch.no_grad():
                #     responses = []
                #     for digit in range(10):
                #         mask = target == digit
                #         if mask.any():
                #             digit_responses = expert_outputs[mask].mean(0)
                #             responses.append(digit_responses.cpu().numpy())
                #     responses = np.array(responses)
                #     print("\nExpert responses per digit:")
                #     print(responses)
    
    return model

def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

def evaluate_experts(model, test_loader, device):
    model.eval()
    expert_responses = torch.zeros(10, 10)  # [digit, expert]
    counts = torch.zeros(10)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, _, expert_outputs = model(data)
            
            for digit in range(10):
                mask = target == digit
                if mask.any():
                    expert_responses[digit] += expert_outputs[mask].sum(0)
                    counts[digit] += mask.sum()
    
    # 计算每个数字的平均专家响应
    for digit in range(10):
        if counts[digit] > 0:
            expert_responses[digit] /= counts[digit]
    
    return expert_responses

def plot_expert_assignments(expert_responses):
    plt.figure(figsize=(10, 8))
    sns.heatmap(expert_responses.cpu().numpy(), 
                xticklabels=range(10), 
                yticklabels=range(10),
                annot=True)
    plt.xlabel('Expert')
    plt.ylabel('Digit')
    plt.title('Expert Response to Each Digit')
    plt.show()

def analyze_prediction(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output, gate_scores, expert_outputs = model(image)
        prediction = output.argmax(dim=1)
        
        print("Predicted digit:", prediction.item())
        print("\nExpert outputs:")
        print(expert_outputs[0].cpu().numpy())
        print("\nGate scores:")
        print(gate_scores[0].cpu().numpy())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练模型
    model = train_digit_moe()
    
    # 评估专家分配
    test_loader = get_test_loader()
    expert_responses = evaluate_experts(model, test_loader, device)
    plot_expert_assignments(expert_responses)
    
    # 分析单个预测示例
    test_loader = get_test_loader()
    test_images, _ = next(iter(test_loader))
    analyze_prediction(model, test_images[0:1], device)

if __name__ == "__main__":
    main()