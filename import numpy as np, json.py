import numpy as np, json
from nn_predict import nn_inference

weights    = np.load(r"C:\Users\henry\Desktop\dnn-inference-113C52010\model\fashion_mnist.npz")
model_arch = json.load(open(r"C:\Users\henry\Desktop\dnn-inference-113C52010\model\fashion_mnist.json"))
# 隨便拿一張測試圖片
x_test, y_test = "C:\Users\henry\Desktop\dnn-inference-113C52010\data\fashion\t10k-images-idx3-ubyte.gz"
x = x_test[0:1] / 255.0
out = nn_inference(model_arch, weights, np.expand_dims(x, -1))
print(out.shape)      # 應該是 (1, 10)
print(np.argmax(out))
print("Label:", y_test[0])