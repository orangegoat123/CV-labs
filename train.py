from data_loader import load_cifar10
from model import ThreeLayerNet
from utils import evaluate, plot_training_curves
import numpy as np
from utils import plot_training_curves, evaluate, plot_learning_rate
def train_model():
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    
    # 模型参数
    input_size = 3072
    hidden1_size = 1024
    hidden2_size = 512
    output_size = 10
    reg = 1e-5
    lr = 5e-4
    batch_size = 256
    num_epochs = 50
    
    # 初始化模型
    model = ThreeLayerNet(
        input_size=input_size,
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
        output_size=output_size,
        reg_lamda=reg,
        activation='relu'
    )
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    original_lr = lr
    
    for epoch in range(num_epochs):
        # 学习率衰减 (cosine)
        lr *= np.cos(7 * np.pi * epoch / (16 * num_epochs))
        current_lr = lr * np.cos(7 * np.pi * epoch / (16 * num_epochs))
        learning_rates.append(current_lr) 
        # Mini-batch训练
        indices = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            # 前向传播
            _ = model.forward(X_batch)
            
            # 反向传播
            grads = model.backward(X_batch, y_batch)
            
            # 带动量的参数更新
            for param in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
                model.velocity[param] = 0.9 * model.velocity[param] - lr * grads[param]
                setattr(model, param, getattr(model, param) + model.velocity[param])
        
        # 计算训练损失和验证集准确率
        train_loss = model.compute_loss(X_train[:10000], y_train[:10000])
        val_loss = model.compute_loss(X_val, y_val)
        val_acc = evaluate(model, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.savez('best_model.npz',
                    W1=model.W1, b1=model.b1,
                    W2=model.W2, b2=model.b2,
                    W3=model.W3, b3=model.b3)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    # 绘制学习率变化曲线    
    plot_learning_rate(learning_rates, num_epochs, initial_lr=original_lr)    
    
    # 可视化训练曲线
    plot_training_curves(train_losses,val_losses,val_accuracies)

if __name__ == '__main__':
    train_model()