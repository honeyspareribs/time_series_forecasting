import torch
import numpy as np
from model import LSTM, TransformerModel
from torch.utils.data import DataLoader
from data import SequenceData
from torch import nn
import matplotlib.pyplot as plt

# 创建一个数据集实例
test_data = SequenceData('./data/test')
test_data_loader = DataLoader(test_data, batch_size=100)

model_type = "LSTM"

# 创建模型实例并设置在GPU上运行
if model_type == "LSTM":
    model = LSTM(512).cuda()
elif model_type == "GPT":
    model = TransformerModel(hidden_size=128, nhead=16).cuda()

# 加载模型
if model_type == "LSTM":
    model.load_state_dict(torch.load('./exp/LSTM/model.pt'))
elif model_type == "GPT":
    model.load_state_dict(torch.load('./exp/GPT/model.pt'))

# 开始测试
model.eval()

predictions = []
actuals = []

criterion = nn.MSELoss()

with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_data_loader):
        # 将数据移到GPU上
        inputs = inputs.cuda()
        targets = targets.cuda()

        # 预测
        outputs, _ = model(inputs)

        loss = criterion(outputs, targets)

        print("loss: ", loss.item())

        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

predictions = np.concatenate(predictions).ravel()
actuals = np.concatenate(actuals).ravel()

plt.plot(predictions, label='Predicted')
plt.plot(actuals, label='Actual')
plt.legend()
plt.savefig("test.png")
