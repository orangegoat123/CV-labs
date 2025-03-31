import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size=3072, hidden1_size=1024,hidden2_size=512, output_size=10, 
                 activation='relu', reg_lamda=1e-5):
        # 参数初始化
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden1_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0/hidden1_size)
        self.b2 = np.zeros(hidden2_size)
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0/hidden2_size)
        self.b3 = np.zeros(output_size)
        self.reg_lamda = reg_lamda
        self.activation = activation
        self.velocity = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2),
            'W3': np.zeros_like(self.W3),
            'b3': np.zeros_like(self.b3)
        }

    def _relu(self, x): return np.maximum(0, x)
    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """前向传播"""
        self.z1 = X.dot(self.W1) + self.b1
        if self.activation == 'relu':
            self.a1 = self._relu(self.z1)
        else:
            self.a1 = self._sigmoid(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        if self.activation =='relu':
            self.a2 = self._relu(self.z2)
        else:
            self.a2 = self._sigmoid(self.z2)
        self.z3 = self.a2.dot(self.W3) + self.b3
        self.probs = self._softmax(self.z3)
        return self.probs
        

    def backward(self, X, y):
        """反向传播"""
        delta4 = self.probs
        delta4[np.arange(len(X)), y] -= 1
        delta4 /= len(X)

        dW3 = self.a2.T.dot(delta4) + self.reg_lamda * self.W3
        db3 = np.sum(delta4, axis=0)

        if self.activation == 'relu':
            delta3 = delta4.dot(self.W3.T) * (self.z2 > 0)
        else:
            delta3 = delta4.dot(self.W3.T) * (self.a2 * (1 - self.a2))

        dW2 = self.a1.T.dot(delta3) + self.reg_lamda * self.W2
        db2 = np.sum(delta3, axis=0)

        if self.activation =='relu':
            delta2 = delta3.dot(self.W2.T) * (self.z1 > 0)
        else:
            delta2 = delta3.dot(self.W2.T) * (self.a1 * (1 - self.a1))
        dW1 = X.T.dot(delta2) + self.reg_lamda * self.W1
        db1 = np.sum(delta2, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def compute_loss(self, X, y):
        """计算损失"""
        probs = self.forward(X)
        corect_logprobs = -np.log(probs[np.arange(len(X)), y])
        data_loss = np.sum(corect_logprobs) / len(X)
        reg_loss = 0.5 * self.reg_lamda * (np.sum(self.W1**2) + np.sum(self.W2**2)+np.sum(self.W3**2))
        return data_loss + reg_loss
