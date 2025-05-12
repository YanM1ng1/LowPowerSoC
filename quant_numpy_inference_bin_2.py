import numpy as np
import gzip
import struct
from torchvision import datasets, transforms

def binarize_input(x):
    return (x > 0.5).float() 

def relu(x):
    return np.maximum(0, x).astype(np.int32)  # 修改為 int32

def maxpool2d(x, size=2):
    N, C, H, W = x.shape
    out = np.zeros((N, C, H // size, W // size), dtype=np.int32)  # 修改為 int32
    for i in range(0, H, size):
        for j in range(0, W, size):
            out[:, :, i // size, j // size] = np.max(x[:, :, i:i+size, j:j+size], axis=(2, 3))
    return out

def conv2d(x, w, b, scale_in, zp_in, scale_w, zp_w, scale_out, zp_out, pad=1):
    N, C, H, W = x.shape
    Cout, Cin, kH, kW = w.shape
    out = np.zeros((N, Cout, H, W), dtype=np.int32)
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=zp_in)

    for n in range(N):
        for c in range(Cout):
            for i in range(H):
                for j in range(W):
                    region = x_padded[n, :, i:i+kH, j:j+kW]
                    # 反量化
                    region_float = (region - zp_in) * scale_in
                    w_float = (w[c] - zp_w) * scale_w
                    # 計算卷積
                    out_float = np.sum(region_float * w_float) + b[c]
                    # 重新量化
                    out[n, c, i, j] = np.round(out_float / scale_out) + zp_out

    return out

def adaptive_avgpool2d(x):
    # 整數平均，使用整數除法
    sum_x = np.sum(x, axis=(2, 3), keepdims=True)  # 保持 int32
    count = x.shape[2] * x.shape[3]
    avg = sum_x // count
    return avg  # 保持 int32

def fc(x, w, b):
    x_int32 = x.astype(np.int32)
    w_int32 = w.astype(np.int32)
    out = np.dot(x_int32, w_int32.T) + b  # 保持 int32
    return out  # 不再 clip，保持 int32
    return out

def load_txt_weights_txt_format(path):
    return np.loadtxt(path, dtype=np.int8)

def load_txt_weights_txt_format_bias(path):
    return np.loadtxt(path, dtype=np.int32)

def main():
    print("載入測試資料...")
    transform = transforms.Compose([
        transforms.CenterCrop(16),
        transforms.ToTensor(),
        transforms.Lambda(binarize_input)
    ])
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    y_test = mnist_test.targets.numpy()
    x_test = np.array([np.array(img[0], dtype=np.int8) for img in mnist_test]).reshape(-1, 1, 16, 16)

    print("載入 INT8 權重...")
    weights = {
        'conv1.weight': load_txt_weights_txt_format("quant_weights_txt/conv1.weight.txt").reshape(6, 1, 3, 3),
        'conv1.bias':   load_txt_weights_txt_format_bias("quant_weights_txt/conv1.bias.txt"),
        'conv1.scale':  0.05,  # 從 scale_zero_point.txt 中讀取
        'conv1.zp':     0,     # 從 scale_zero_point.txt 中讀取
        'conv2.weight': load_txt_weights_txt_format("quant_weights_txt/conv2.weight.txt").reshape(16, 6, 3, 3),
        'conv2.bias':   load_txt_weights_txt_format_bias("quant_weights_txt/conv2.bias.txt"),
        'conv2.scale':  0.05,  # 從 scale_zero_point.txt 中讀取
        'conv2.zp':     0,     # 從 scale_zero_point.txt 中讀取
        'fc.weight':    load_txt_weights_txt_format("quant_weights_txt/fc.weight.txt").reshape(10, 16),
        'fc.bias':      load_txt_weights_txt_format_bias("quant_weights_txt/fc.bias.txt"),
        'fc.scale':     0.05,  # 從 scale_zero_point.txt 中讀取
        'fc.zp':        0,     # 從 scale_zero_point.txt 中讀取
        
    }

    print("推論中...")
    x = conv2d(x_test[:3], weights['conv1.weight'], weights['conv1.bias'],
               scale_in=1.0, zp_in=0,
               scale_w=weights['conv1.scale'], zp_w=weights['conv1.zp'],
               scale_out=0.1, zp_out=0)
    x = relu(x)
    x = maxpool2d(x)
    # 其他層類似處理
    x = conv2d(x, weights['conv2.weight'], weights['conv2.bias'], pad=1)
    x = relu(x)
    print(x[:1])
    input()
    x = maxpool2d(x)
    x = adaptive_avgpool2d(x)
    print(x[:1])
    input()
    x = x.reshape(x.shape[0], -1)
    x = fc(x, weights['fc.weight'], weights['fc.bias'])
    print(x[:1])
    input()
    preds = np.argmax(x, axis=1)
    acc = np.mean(preds == y_test[:3])
    print(f"[INT8 NumPy 推論（全整數）] Accuracy on 5000 samples: {acc:.4f}")

if __name__ == "__main__":
    main()
