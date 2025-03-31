import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# CIFAR-10官方统计数据（每个通道的均值和标准差）
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])

def augment_image(images):
    """数据增强：随机水平翻转 + 随机裁剪（保持展平后格式）"""
    # 将图像恢复为32x32x3格式
    images = images.reshape(-1, 32, 32, 3)
    augmented = []
    
    for img in images:
        # 随机水平翻转
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]
        
        # 随机裁剪（填充后裁剪）
        pad = 4
        padded = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='constant')
        h_start = np.random.randint(0, 2*pad)
        w_start = np.random.randint(0, 2*pad)
        cropped = padded[h_start:h_start+32, w_start:w_start+32, :]
        
        augmented.append(cropped.reshape(-1))
    
    return np.array(augmented)

def load_cifar10(augment=True):
    """加载并预处理CIFAR-10数据集"""
    def _load_file(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']
    
    # 加载原始数据
    X_train, y_train = [], []
    for i in range(1, 6):
        data, labels = _load_file(f'../data/cifar-10-batches-py/data_batch_{i}')
        X_train.append(data)
        y_train.extend(labels)
    X_train = np.vstack(X_train).astype(np.float32)
    y_train = np.array(y_train)
    
    X_test, y_test = _load_file('../data/cifar-10-batches-py/test_batch')
    X_test = X_test.astype(np.float32)
    y_test = np.array(y_test)
    
    # 将每个通道的统计量扩展到所有像素
    # 原始数据格式为[RGB通道][像素]，展平后结构为R...R G...G B...B
    n_pixels = 32 * 32
    mean_expanded = np.repeat(CIFAR10_MEAN, n_pixels)
    std_expanded = np.repeat(CIFAR10_STD, n_pixels)
    
    # 归一化处理
    X_train = (X_train / 255.0 - mean_expanded) / std_expanded
    X_test = (X_test / 255.0 - mean_expanded) / std_expanded
    
    # 划分验证集（验证集不需要增强）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # 数据增强
    if augment:
        X_aug = augment_image(X_train)
        X_train = np.vstack([X_train, X_aug])
        y_train = np.concatenate([y_train, y_train])
    
    return X_train, y_train, X_val, y_val, X_test, y_test