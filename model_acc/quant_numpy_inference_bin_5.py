import numpy as np
from torchvision import datasets, transforms



def binarize_input(x):
    return (x > 0.5).float()

def relu(x):
    return np.maximum(0, x).astype(np.int8)

def maxpool2d(x, size=2):
    N, C, H, W = x.shape
    out = np.zeros((N, C, H // size, W // size), dtype=np.int8)
    for i in range(0, H, size):
        for j in range(0, W, size):
            out[:, :, i // size, j // size] = np.max(x[:, :, i:i+size, j:j+size], axis=(2, 3))
    return out

def adaptive_avgpool2d(x):
    sum_x = np.sum(x.astype(np.int32), axis=(2, 3), keepdims=True)
    count = x.shape[2] * x.shape[3]
    avg = sum_x // count
    return np.clip(avg, -128, 127).astype(np.int8)

def quantize_scale(scale_float, bit_width=32):
    """
    將 scale_float (任意正數) 轉為整數乘法器 + 右移位數（不再限制 <1.0）
    """
    assert scale_float > 0, "scale 必須是正數"
    max_int = 2 ** (bit_width//4) - 1

    shift = 0
    while (scale_float * (1 << shift)) < max_int and shift < bit_width:
        shift += 1

    M = int(round(scale_float * (1 << shift)))
    max_int = 2 ** (bit_width) - 1
    if M >= max_int:
        shift += 1
        M = int(round(scale_float * (1 << shift)))
        if M >= max_int:
            raise ValueError("scale_float 太大，無法轉為 (M, shift)")

    return M, shift


def quantized_conv2d_binary_input_precise(x, w_int8, b_float32, scale_x, scale_w, scale_y, pad=1, boost_bits=4):


    global check

    N, C, H, W = x.shape
    Cout, cc, kH, kW = w_int8.shape

    out = np.zeros((N, Cout, H, W), dtype=np.int8)
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    bias_int32 = np.round(b_float32 / scale_y).astype(np.int32)

    # if check == 1:
        # print(w_int8[0,:, :, :])
        # input("conv2d input:")
    #     print('x_padded = ', x_padded)
    #     print('bias_int32 = ', bias_int32)
    #     input("conv2d input:")
    for n in range(N):
        for c in range(Cout):
            w_c = w_int8[c].astype(np.int32)
            # 放大比例（左移 boost_bits 位元 = * 2^boost_bits）
            inverse_scale_y_int = int(1 / scale_y)
            M_float = (scale_x * scale_w[c] * inverse_scale_y_int) * (1 << boost_bits)
            # print('scale_w = ', scale_w[c])
            # print('scale_y = ', scale_y)
            # print('M_float = ', M_float)
            M_int, shift = quantize_scale(M_float)
            for i in range(H):
                for j in range(W):
                    acc = 0  # 初始化加總值（int32）

                    for c_in in range(cc):  # 每個輸入通道個別處理
                        region = x_padded[n, c_in, i:i + kH, j:j + kW].astype(np.int32)
                        kernel = w_c[c_in]  # shape: (3, 3)
                        partial = np.sum(region * kernel)

                        # 個別 kernel 做 scaling + shift + clip
                        partial = ((partial * M_int) >> shift) >> boost_bits
                        # partial = np.clip(partial, -128, 127)

                        acc += partial  # 加總每個 kernel 的結果
                        # if check == 1 and n ==0 and c ==1 and i ==7:
                        #     print('partial = ', partial)
                        #     print('acc = ', acc)    
                        #     input("cp")
                    # acc = ((acc * M_int) >> shift) >> boost_bits
                    # 加上 bias
                    acc += bias_int32[c]

                    # 最終 clip 並寫入
                    out[n, c, i, j] = np.clip(acc, -128, 127)
                    

                    # if check == 1 and n ==0 and c ==1 and i ==7:
                    #     print('x_padded[n, :, i:i + kH, j:j + kW] = ', x_padded[n, :, i:i + kH, j:j + kW])
                    #     print('region * w_c = ', region * w_c)
                    #     print('w_c = ', w_c)
                    #     print('acc = ', acc)
                    #     print('bias_int32[c] = ', bias_int32[c])
                    #     print('M_int = ', M_int)
                    #     print('shift = ', shift)
                    #     print('boost_bits = ', boost_bits)
                    #     input("conv2d output111:")

    return out


def quantized_fc(x, w_int8, b_float32, scale_x, scale_w, scale_y):
    N, in_feat = x.shape
    out_feat = w_int8.shape[0]
    out = np.zeros((N, out_feat), dtype=np.int8)
    bias_int32 = np.round(b_float32 / scale_y).astype(np.int32)
    for i in range(out_feat):
        M_float = (scale_x * scale_w[i]) / scale_y
        M_int, shift = quantize_scale(M_float)
        acc = np.dot(x.astype(np.int32), w_int8[i].astype(np.int32)) + bias_int32[i]
        acc = (acc * M_int) >> shift
        out[:, i] = np.clip(acc, -128, 127)
    return out

def load_txt_weights_txt_format(path):
    return np.loadtxt(path, dtype=np.int8)

def load_txt_weights_txt_format_bias(path):
    return np.loadtxt(path, dtype=np.int32)

def load_txt_weights_txt_format_float(path):
    return np.loadtxt(path, dtype=np.float32)



def main():
    transform = transforms.Compose([
        transforms.CenterCrop(16),
        transforms.ToTensor(),
        transforms.Lambda(binarize_input)
    ])
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    y_test = mnist_test.targets.numpy()
    x_test = np.array([np.array(img[0], dtype=np.int8) for img in mnist_test]).reshape(-1, 1, 16, 16)
    


    weights = {}
    weights['conv1.weight'] = load_txt_weights_txt_format("quant_weights_txt/conv1_int8.txt").reshape(6, 1, 3, 3)
    weights['conv1.bias'] = load_txt_weights_txt_format_float("quant_weights_txt/conv1_bias.txt")
    weights['conv1.scale'] = load_txt_weights_txt_format_float("quant_weights_txt/conv1_scale_bias.txt")
    weights['conv2.weight'] = load_txt_weights_txt_format("quant_weights_txt/conv2_int8.txt").reshape(12, 6, 3, 3)
    weights['conv2.bias'] = load_txt_weights_txt_format_float("quant_weights_txt/conv2_bias.txt")
    weights['conv2.scale'] = load_txt_weights_txt_format_float("quant_weights_txt/conv2_scale_bias.txt")
    weights['fc.weight'] = load_txt_weights_txt_format("quant_weights_txt/fc_int8.txt").reshape(10, 12)
    weights['fc.bias'] = load_txt_weights_txt_format_bias("quant_weights_txt/fc_bias.txt")
    weights['fc.scale'] = load_txt_weights_txt_format_float("quant_weights_txt/fc_scale_bias.txt")

    # 各層輸出 scale（你可根據模型實際值修改）
    scale_conv1_out = 0.0412
    scale_conv2_out = 0.11
    scale_fc_out = 0.2028

    x = x_test[:1000 ]


    x = quantized_conv2d_binary_input_precise(x, weights['conv1.weight'], weights['conv1.bias'],1,
                                      weights['conv1.scale'], scale_conv1_out)


    x = relu(x)
    # print('x[0]')
    # print(x[0])
    # input('x[0] conv1 output:') 
    # print('before maxpool2d')
    # print(x[1])
    # input("co11nv1 output:")
    x = maxpool2d(x)
    # print('after maxpool2d')

    # print('x[0]')

    # print(weights['conv2.weight'])
    # input("conv2d output:")
    global check
    check = 1

    x = quantized_conv2d_binary_input_precise(x, weights['conv2.weight'], weights['conv2.bias'],scale_conv1_out,
                                      weights['conv2.scale'], scale_conv2_out)
    x = relu(x)

    print(x[0])
    input("maxpool2d output:")

    x = maxpool2d(x)
    print(x[0])
    x = maxpool2d(x, size=4)
    # x = adaptive_avgpool2d(x)
    print(x[0])
    input('before flatten:')
    # x = maxpool2d(x, size=2)
    x = x.reshape(x.shape[0], -1)


    x = quantized_fc(x, weights['fc.weight'], weights['fc.bias'], scale_conv2_out,
                     weights['fc.scale'], scale_fc_out)
    print(x[0])
    input('before softmax:')
    preds = np.argmax(x, axis=1)
    acc = np.mean(preds == y_test[:1000])
    print(f"[INT8 推論] Accuracy on first 1000 samples: {acc:.4f}")
    # for i in range(100):
    #     print(f"  Sample {i}: Prediction={preds[i]}, Label={y_test[i]}")

if __name__ == "__main__":
    global check
    check = 0
    main()
