import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化多层感知机
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层神经元数量
        :param output_size: 输出层神经元数量
        :param learning_rate: 学习率
        """
        # 初始化权重
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
        
        # 学习率
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """sigmoid函数的导数"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        前向传播
        :param X: 输入数据，形状为(n_samples, input_size)
        :return: 输出层的预测结果
        """
        # 隐藏层计算
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)  # 隐藏层激活值
        
        # 输出层计算
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)  # 输出层激活值
        
        return self.a2
    
    def compute_loss(self, y_pred, y_true):
        """
        计算交叉熵损失
        :param y_pred: 预测值
        :param y_true: 真实标签
        :return: 损失值
        """
        m = y_true.shape[0]
        loss = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, X, y_true, y_pred):
        """
        反向传播，计算梯度并更新权重
        :param X: 输入数据
        :param y_true: 真实标签
        :param y_pred: 预测值
        """
        m = y_true.shape[0]
        
        # 计算输出层的误差和梯度
        delta2 = y_pred - y_true
        d_weights2 = np.dot(self.a1.T, delta2) / m
        d_bias2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # 计算隐藏层的误差和梯度
        delta1 = np.dot(delta2, self.weights2.T) * self.sigmoid_derivative(self.a1)
        d_weights1 = np.dot(X.T, delta1) / m
        d_bias1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1
        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        训练模型
        :param X: 输入数据
        :param y: 标签
        :param epochs: 训练轮数
        :param verbose: 是否打印训练过程
        """
        for i in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y_pred, y)
            
            # 反向传播和参数更新
            self.backward(X, y, y_pred)
            
            # 打印训练信息
            if verbose and (i % 100 == 0):
                print(f"Epoch {i}, Loss: {loss:.4f}")
    
    def predict(self, X, threshold=0.5):
        """
        预测函数
        :param X: 输入数据
        :param threshold: 分类阈值
        :return: 预测类别
        """
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)


# 测试MLP
if __name__ == "__main__":
    # 创建简单的二分类数据集 (XOR问题)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR的输出
    
    # 创建并训练MLP
    mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    mlp.train(X, y, epochs=10000, verbose=True)
    
    # 测试预测
    predictions = mlp.predict(X)
    print("\n预测结果:")
    for i in range(len(X)):
        print(f"输入: {X[i]}, 预测: {predictions[i][0]}, 实际: {y[i][0]}")
