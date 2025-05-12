import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np
import os

# --------------------------
# 1. 定義 CNN 模型（支援 QAT）
# --------------------------
class SmallQuantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.gap = nn.MaxPool2d(4)
        self.fc  = nn.Linear(12, 10)

        torch.quantization.fuse_modules(self, ['conv1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'relu2'], inplace=True)

    def forward(self, x):
        x = self.quant(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# 匯出為 .txt
# def export_quantized_weights_to_txt(model, dir_path="quant_weights_txt"):
#     print("导出 INT8 权重为 .txt...")
#     os.makedirs(dir_path, exist_ok=True)
#     state_dict = model.state_dict()
#     for k, v in state_dict.items():
#         if not isinstance(v, torch.Tensor):
#             continue
#         if v.is_quantized:
#             arr = v.dequantize().cpu().numpy()
#         else:
#             arr = v.detach().cpu().numpy()
#         scale = np.max(np.abs(arr)) + 1e-8
#         arr_int8 = np.clip((arr / scale) * 127, -128, 127).astype(np.int8)
#         txt_path = os.path.join(dir_path, f"{k}.txt")
#         np.savetxt(txt_path, arr_int8.flatten(), fmt="%d")
#     print(f"已將 INT8 權重儲存為個別 .txt 檔案，位於資料夾: {dir_path}")

def export_quantized_weights_to_txt(model, dir_path="quant_weights_txt"):
    print("导出 INT8 权重为 .txt...")
    os.makedirs(dir_path, exist_ok=True)
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.is_quantized:
            # 提取量化權重的整數值、比例和零點
            arr_int8 = v.int_repr().cpu().numpy()  # INT8 整數值
            scale = v.q_scale()  # 量化比例
            zero_point = v.q_zero_point()  # 零點
            # 儲存 INT8 權重
            txt_path = os.path.join(dir_path, f"{k}.weight.txt")
            np.savetxt(txt_path, arr_int8.flatten(), fmt="%d")
            # 儲存比例和零點
            with open(os.path.join(dir_path, f"{k}.scale_zero_point.txt"), "w") as f:
                f.write(f"{scale}\n{zero_point}\n")
        else:
            # 偏置直接儲存為浮點數
            arr = v.detach().cpu().numpy()
            txt_path = os.path.join(dir_path, f"{k}.bias.txt")
            np.savetxt(txt_path, arr.flatten(), fmt="%.6f")
    print(f"已將 INT8 權重儲存為個別 .txt 檔案，位於資料夾: {dir_path}")


# --------------------------
# 2. 訓練 / 驗證 函數
# --------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

# --------------------------
# 3. 主程序
# --------------------------
def binarize_input(x):
    return (x > 0.5).float()

def main():
    multiprocessing.freeze_support()
    torch.backends.quantized.engine = 'fbgemm'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用裝置:", device)

    transform = transforms.Compose([
        transforms.CenterCrop(16),
        transforms.ToTensor(),
        transforms.Lambda(binarize_input)
    ])

    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = SmallQuantCNN().to(device)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, 21):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, test_loader, criterion, device)
        print(f'Epoch {epoch:2d} | Train Acc: {tr_acc:.3f} | Val Acc: {val_acc:.3f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_qat.pth')
        if best_acc >= 0.90:
            print("已達 90% 準確度，提前停止訓練")
            break

    # --- 先匯出 float32 權重 (.txt) ---
    # export_quantized_weights_to_txt(model)    # model 還是 float32

    # --- 再量化並儲存 QAT 模型 ---
    print("轉換為 INT8 模型...")
    model.cpu().eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    torch.save(quantized_model.state_dict(), 'mnist_int8_2.pth')
    # --- 匯出 INT8 權重 (.txt) ---
    export_quantized_weights_to_txt(quantized_model)


    test_loss, test_acc = eval_model(quantized_model, test_loader, criterion, torch.device('cpu'))
    print(f'量化後測試準確度: {test_acc:.3f}')

if __name__ == "__main__":
    main()
