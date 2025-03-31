from data_loader import load_cifar10
from model import ThreeLayerNet
import numpy as np
from utils import evaluate
def hyperparameter_search():
    X_train, y_train, X_val, y_val, _, _ = load_cifar10(augment=False)
    
    # 搜索空间
    hidden1_sizes = [512, 1024]
    hidden2_sizes = [128, 256]
    learning_rates = [1e-3, 5e-4]
    reg_strengths = [1e-5, 1e-4]
    
    best_acc = 0
    best_params = {}
    
    for hidden1_size in hidden1_sizes:
        for hidden2_size in hidden2_sizes:
            for lr in learning_rates:
                for reg in reg_strengths:
                    print(f'Trying hidden1_size={hidden1_size},hidden2_size={hidden2_size}, lr={lr}, reg={reg}')
                    model = ThreeLayerNet(
                        input_size=3072,
                        hidden1_size=hidden1_size,
                        hidden2_size=hidden2_size,
                        output_size=10,
                        reg_lamda=reg
                )
                
                    # 简化的训练循环
                    for epoch in range(10):
                        indices = np.random.permutation(X_train.shape[0])
                        for i in range(0, X_train.shape[0], 256):
                            batch_idx = indices[i:i+256]
                            X_batch = X_train[batch_idx]
                            y_batch = y_train[batch_idx]
                        
                            # 前向传播
                            _ = model.forward(X_batch)
                            # 反向传播
                            grads = model.backward(X_batch, y_batch)
                            # 参数更新
                            for param in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
                                model.velocity[param] = 0.9 * model.velocity[param] - lr * grads[param]
                                setattr(model, param, getattr(model, param) + model.velocity[param])
                
                    val_acc = evaluate(model, X_val, y_val)
                    print(f'Validation Accuracy: {val_acc:.4f}')
                
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_params = {
                            'hidden1_size': hidden1_size,
                            'hidden2_size': hidden2_size,
                            'lr': lr,
                            'reg': reg
                        }
    
    print('\nBest Parameters:')
    print(best_params)
    return best_params

if __name__ == '__main__':
    hyperparameter_search()