# inference_test.py
import numpy as np
import json
from nn_predict import nn_inference, softmax
from utils import mnist_reader

def run_inference_test(num_samples=100):
    # 1. 載入原始 idx 測試集
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    
    # 2. 讀取你的模型架構與權重
    weights    = np.load('model/fashion_mnist.npz')
    with open('model/fashion_mnist.json', 'r', encoding='utf-8') as f:
        model_arch = json.load(f)
    
    # 3. 前處理：歸一化並加入 channel 維度
    x = x_test[:num_samples].astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)            # (num_samples,28,28,1)

    # 4. 推論
    logits = nn_inference(model_arch, weights, x)  # (num_samples,10)
    probs  = softmax(logits)                       # (num_samples,10)

    # 5. 計算準確度
    preds = np.argmax(probs, axis=1)
    acc   = np.mean(preds == y_test[:num_samples])

    print(f"前 {num_samples} 筆測試資料準確率：{acc * 100:.2f}%")

if __name__ == "__main__":
    run_inference_test(1000)
