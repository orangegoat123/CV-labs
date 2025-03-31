import matplotlib.pyplot as plt
import numpy as np
from model import ThreeLayerNet
def visualize_weight_matrices(model, save_path='weight_matrices.png'):
    """可视化三个权重矩阵的结构
    
    参数:
        model (ThreeLayerNet): 训练好的模型实例
        save_path (str): 图像保存路径
    """
    plt.figure(figsize=(18, 5))
    
    # ----------------------
    # W1矩阵（输入层→隐藏层1）
    # ----------------------
    plt.subplot(131)
    w1_subset = model.W1[:, :128]  # 取前128个神经元
    im1 = plt.imshow(w1_subset.T, cmap='coolwarm', aspect='auto',
                   vmin=-0.1, vmax=0.1)
    plt.title("(a) W1 Matrix (First 128 Neurons)", fontsize=12)
    plt.xlabel("Input Pixels (3072-d)")
    plt.ylabel("Hidden Layer 1 Neurons (First 128)")
    plt.colorbar(im1, fraction=0.046, pad=0.04)

    # ----------------------
    # W2矩阵（隐藏层1→隐藏层2）
    # ----------------------
    plt.subplot(132)
    w2_subset = model.W2[:128, :128]  # 取128x128子集
    im2 = plt.imshow(w2_subset.T, cmap='coolwarm', aspect='auto',
                   vmin=-0.05, vmax=0.05)
    plt.title("(b) W2 Matrix (128x128 Subset)", fontsize=12)
    plt.xlabel("Hidden Layer 1 Neurons (First 128)")
    plt.ylabel("Hidden Layer 2 Neurons (First 128)")
    plt.colorbar(im2, fraction=0.046, pad=0.04)

    # ----------------------
    # W3矩阵（隐藏层2→输出层）
    # ----------------------
    plt.subplot(133)
    w3_matrix = model.W3.T  # 转置为[10, 512]
    im3 = plt.imshow(w3_matrix, cmap='coolwarm', aspect='auto',
                   vmin=-0.02, vmax=0.02)
    plt.title("(c) W3 Output Layer Matrix", fontsize=12)
    plt.xlabel("Hidden Layer 2 Neurons (512-d)")
    plt.ylabel("Output Classes (10-d)")
    plt.yticks(range(10), range(10))
    plt.colorbar(im3, fraction=0.046, pad=0.04)

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 使用示例 --------------------------------------------------
if __name__ == '__main__':
    # 加载训练好的模型
    model = ThreeLayerNet()
    loaded = np.load('best_model.npz')
    model.W1 = loaded['W1']
    model.W2 = loaded['W2']
    model.W3 = loaded['W3']
    
    # 生成可视化
    visualize_weight_matrices(model, 'weight_matrices.png')