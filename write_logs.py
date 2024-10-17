import torch
import torch.nn as nn
import torch.optim as optim

# Kiểm tra GPU
# device = torch.device("cpu")
device = torch.device("cuda")
# Định nghĩa mô hình đơn giản


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Tạo mô hình và chuyển sang GPU nếu có
model = SimpleModel().to(device)

# Tạo dữ liệu giả lập
data = torch.randn(64, 10).to(device)  # Dữ liệu chuyển sang GPU
target = torch.randn(64, 1).to(device)

# Khởi tạo loss và optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop đơn giản
for epoch in range(10):
    optimizer.zero_grad()          # Xóa gradient trước đó
    output = model(data)           # Forward pass trên GPU
    loss = criterion(output, target)  # Tính loss
    loss.backward()                # Backward pass trên GPU
    optimizer.step()               # Cập nhật trọng số

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
