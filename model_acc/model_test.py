import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 🧠 定義模型架構（與你訓練用的完全一樣）
class SmallQuantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.gap = nn.MaxPool2d(4)
        self.fc  = nn.Linear(12, 10)

        # 模組 fuse 對 QAT 模型匯入很重要（需要與訓練一致）
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

# ✅ 資料前處理
def binarize_input(x):
    return (x > 0.5).float()

transform = transforms.Compose([
    transforms.CenterCrop(16),
    transforms.ToTensor(),
    transforms.Lambda(binarize_input)
])

# ✅ 載入測試資料集
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# ✅ 建立模型並量化
model = SmallQuantCNN()
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)  # 注意：此處使用 PTQ prepare
quantized_model = torch.quantization.convert(model, inplace=False)

# ✅ 載入你已經轉換好的量化權重
quantized_model.load_state_dict(torch.load("mnist_int8_2.pth", map_location='cpu'))
quantized_model.eval()

# ✅ 計算準確度
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = quantized_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"量化模型推論準確率：{acc:.2f}%")


import torch

# 儲存中間輸出
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 建立模型並載入量化權重
model = SmallQuantCNN()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
quantized_model = torch.quantization.convert(model, inplace=False)
quantized_model.load_state_dict(torch.load("mnist_int8_2.pth", map_location="cpu"))
quantized_model.eval()

# 加入 hook
quantized_model.conv1.register_forward_hook(get_activation('conv1'))
quantized_model.conv2.register_forward_hook(get_activation('conv2'))
quantized_model.gap.register_forward_hook(get_activation('gap'))
quantized_model.fc.register_forward_hook(get_activation('fc'))


# 準備資料
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def binarize_input(x):
    return (x > 0.5).float()

transform = transforms.Compose([
    transforms.CenterCrop(16),
    transforms.ToTensor(),
    transforms.Lambda(binarize_input)
])

test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1, shuffle=False)
i=0
# 推論單一樣本並顯示中間輸出
with torch.no_grad():
    for images, labels in test_loader:
        outputs = quantized_model(images)
        pred = outputs.argmax(dim=1)
        print(f"預測: {pred.item()}，真實: {labels.item()}")
        print(quantized_model.conv1.weight())
        input()

        print("Conv1 Output INT8:\n", activations['conv1'].int_repr().squeeze().numpy())
        input('conv1 output:')
        print('images:', images)
        print("Conv1 Weights INT8:\n", quantized_model.conv1.weight().int_repr().squeeze().numpy())
        input('conv2 weight:')
        print("Conv2 Weight Scales:\n", quantized_model.conv2.weight().q_per_channel_scales().numpy())
        print("Conv2 Weight ZPs:\n", quantized_model.conv2.weight().q_per_channel_zero_points().numpy())
        print("Conv2 Bias:\n", quantized_model.conv2.bias().detach().numpy())
        print("Conv2 Output INT8:\n", activations['conv2'].int_repr().squeeze().numpy())
        # print("Conv2 Output Scale:\n", activations['conv2'].q_scale())
        # print("Conv2 Output ZP:\n", activations['conv2'].q_zero_point())
        input('conv2 output:')



        print(images)
        
        print("conv1 INT8 值:", activations['conv1'].int_repr().squeeze().numpy())
        print("scale:", activations['conv1'].q_scale())
        print("zero_point:", activations['conv1'].q_zero_point())
        print("conv1 kernel INT8 值:", quantized_model.conv1.weight().int_repr().squeeze().numpy())
        print("conv1 kernel scale:", quantized_model.conv1.weight().q_per_channel_scales().numpy())
        print("conv1 kernel zero_point:", quantized_model.conv1.weight().q_per_channel_zero_points().numpy())
        # print(state_dict["conv1.scale"])

        # print("conv1 bias INT32 值:", quantized_model.conv1.bias().detach().numpy())

        print('conv1 權重:', quantized_model.conv1.weight().int_repr().squeeze().numpy())
        print('conv2 輸入', activations['conv2'])
        print("conv2 INT8 值:", activations['conv2'].int_repr().squeeze().numpy())
        print("scale:", activations['conv2'].q_scale())
        print("zero_point:", activations['conv2'].q_zero_point())
        print("conv2 kernel INT8 值:", quantized_model.conv2.weight().int_repr().squeeze().numpy())
        # print("conv2 kernel scale:", quantized_model.conv2.weight().q_per_channel_scales().numpy())
        input('ckeck pool2d')
        # print("conv2 每個 output channel 對應第 0 個 input channel 的 kernel：")
        # print(quantized_model.conv2.weight().int_repr()[:, 0, :, :].numpy())  
        print("pool1 輸出:", activations['conv2'].int_repr().squeeze().numpy())
        print('activations = ', activations)
        if i ==1 :
            input('123')
        print("gap INT8 值:", activations['gap'].int_repr().squeeze().numpy())
        print("scale:", activations['gap'].q_scale())
        print("zero_point:", activations['gap'].q_zero_point())

        print("fc INT8 值:", activations['fc'].int_repr().squeeze().numpy())
        input('fc output:')
        print("scale:", activations['fc'].q_scale())
        print("zero_point:", activations['fc'].q_zero_point())

        i= i + 1

        # break  # 只看一張圖

input()



# import os
# import numpy as np
# import torch
# def export_quantized_weights_to_txt(model, dir_path="quant_weights_txt"):
#     print("🔍 導出量化模型參數為 .txt ...")
#     os.makedirs(dir_path, exist_ok=True)
#     state_dict = model.state_dict()

#     for k, v in state_dict.items():
#         print(f"🔍 處理 {k} ...")

#         # ✅ 特別處理 packed fc 層（一定要先判斷）
#         if "fc._packed_params._packed_params" in k:
#             print("⚠️ 特別處理 packed fc 層 ...")
#             weight, bias = v  # v 是 tuple: (weight_tensor, bias_tensor)

#             arr_int8 = weight.int_repr().cpu().numpy()
#             np.savetxt(os.path.join(dir_path, "fc_int8.txt"), arr_int8.flatten(), fmt="%d")
#             np.savetxt(os.path.join(dir_path, "fc_bias.txt"), bias.detach().cpu().numpy().flatten(), fmt="%.6f")

#             qscheme = weight.qscheme()
#             if qscheme == torch.per_tensor_affine:
#                 scale = weight.q_scale()
#                 zero_point = weight.q_zero_point()
#                 np.savetxt(os.path.join(dir_path, "fc_scale_bias.txt"), [scale], fmt="%.8f")
#                 np.savetxt(os.path.join(dir_path, "fc_zero_point_bias.txt"), [zero_point], fmt="%d")
#             elif qscheme == torch.per_channel_affine:
#                 scales = weight.q_per_channel_scales().cpu().numpy()
#                 zero_points = weight.q_per_channel_zero_points().cpu().numpy()
#                 axis = weight.q_per_channel_axis()
#                 np.savetxt(os.path.join(dir_path, "fc_scale_bias.txt"), scales, fmt="%.8f")
#                 np.savetxt(os.path.join(dir_path, "fc_zero_point_bias.txt"), zero_points, fmt="%d")
#                 with open(os.path.join(dir_path, "fc_axis.txt"), "w") as f:
#                     f.write(f"{axis}\n")
#             else:
#                 print(f"⚠️ 不支援的 fc qscheme: {qscheme}")
#             continue  # ✅ 已處理完 fc，跳過以下流程

#         # ✅ 跳過非 Tensor 的一般情況
#         if not isinstance(v, torch.Tensor):
#             continue

#         # ✅ 處理 conv1、conv2 等常規 Tensor 權重或 bias
#         base_name = k.replace(".weight", "").replace(".bias", "")

#         if v.is_quantized:
#             arr_int8 = v.int_repr().cpu().numpy()
#             qscheme = v.qscheme()

#             if qscheme == torch.per_tensor_affine:
#                 scale = v.q_scale()
#                 zero_point = v.q_zero_point()
#                 np.savetxt(os.path.join(dir_path, f"{base_name}_int8.txt"), arr_int8.flatten(), fmt="%d")
#                 np.savetxt(os.path.join(dir_path, f"{base_name}_scale_bias.txt"), [scale], fmt="%.8f")
#                 np.savetxt(os.path.join(dir_path, f"{base_name}_zero_point_bias.txt"), [zero_point], fmt="%d")

#             elif qscheme == torch.per_channel_affine:
#                 scales = v.q_per_channel_scales().cpu().numpy()
#                 zero_points = v.q_per_channel_zero_points().cpu().numpy()
#                 axis = v.q_per_channel_axis()
#                 np.savetxt(os.path.join(dir_path, f"{base_name}_int8.txt"), arr_int8.flatten(), fmt="%d")
#                 np.savetxt(os.path.join(dir_path, f"{base_name}_scale_bias.txt"), scales, fmt="%.8f")
#                 np.savetxt(os.path.join(dir_path, f"{base_name}_zero_point_bias.txt"), zero_points, fmt="%d")
#                 with open(os.path.join(dir_path, f"{base_name}_axis.txt"), "w") as f:
#                     f.write(f"{axis}\n")

#             else:
#                 print(f"⚠️ 不支援的量化方式：{qscheme}，跳過 {k}")

#         else:
#             # 非量化 Tensor，例如 bias
#             arr = v.detach().cpu().numpy()
#             np.savetxt(os.path.join(dir_path, f"{base_name}_bias.txt"), arr.flatten(), fmt="%.6f")

#     print(f"✅ 匯出完成：所有量化參數已儲存到 {dir_path}/")

# # ✅ 檢查 conv1 權重的量化方式
# w = quantized_model.conv1.weight()
# print("conv1 qscheme:", w.qscheme())  # 會印出 torch.per_channel_affine 或 torch.per_tensor_affine

# export_quantized_weights_to_txt(quantized_model, "quant_weights_txt")
