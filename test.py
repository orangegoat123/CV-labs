from data_loader import load_cifar10
from model import ThreeLayerNet
import numpy as np
from utils import evaluate
def test_model(model_path='best_model.npz'):
    # 加载数据
    _, _, _, _, X_test, y_test = load_cifar10(augment=False)
    
    # 初始化模型
    model = ThreeLayerNet(input_size=3072, hidden1_size=1024,hidden2_size=512, output_size=10)
    params = np.load(model_path)
    
    # 加载权重
    model.W1 = params['W1']
    model.b1 = params['b1']
    model.W2 = params['W2']
    model.b2 = params['b2']
    model.W3 = params['W3']
    model.b3 = params['b3']
    
    # 评估测试集
    test_acc = evaluate(model, X_test, y_test)
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    test_model()