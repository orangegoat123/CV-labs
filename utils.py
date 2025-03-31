import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, X, y):
    """计算分类准确率"""
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)

def plot_training_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(15, 5))
    
    # 损失曲线子图
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    
    # 准确率曲线子图
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def plot_learning_rate(learning_rates, num_epochs, initial_lr, save_path='lr_curve.png'):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(10, 6))
    epochs = np.arange(num_epochs)
    

    plt.plot(epochs, learning_rates, 
             color='#1f77b4', linewidth=2, 
             marker='o', markersize=6, markevery=5)
    
    plt.title(f"Learning Rate Schedule\n(Initial LR={initial_lr:.1e})", fontsize=14)
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, num_epochs-1)
    
 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()