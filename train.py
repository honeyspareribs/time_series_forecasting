import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import SequenceData
from model import LSTM, TransformerModel

# 创建一个数据集实例
data = SequenceData('./data/train')
data_loader = DataLoader(data, batch_size=20, shuffle=True)

model_type = "LSTM"
# model_type = "GPT"

# 创建模型实例并设置在GPU上运行
if model_type == "LSTM":
    model = LSTM(hidden_dim=512).cuda()
elif model_type == "GPT":
    model = TransformerModel(hidden_size=128, nhead=16).cuda()

# 设置优化器
if model_type == "LSTM":
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif model_type == "GPT":
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 设置损失函数
criterion = torch.nn.MSELoss(reduction='sum')

# 开始训练
for epoch in range(10):
    for i, (inputs, targets) in enumerate(data_loader):
        # 将数据移到GPU上
        inputs = inputs.cuda()
        targets = targets.cuda()

        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs, _ = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss.item()))

# 保存模型
if model_type == "LSTM":
    torch.save(model.state_dict(), './exp/LSTM/model.pt')
elif model_type == "GPT":
    torch.save(model.state_dict(), './exp/GPT/model.pt')
