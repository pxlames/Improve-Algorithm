'''
请在仅依赖 numpy / pandas / scikit-learn 的前提下，完成一个基于 DBSCAN 的异常检测器，并对给定测试样本输出聚类结果（正常簇 0, 1, 2 …；离群点 -1）。



1. 读取数据

    •    train 字段：二维列表，元素均为数值特征

    •    test  字段：二维列表，特征维度与 train 相同

    •    无标签，全为无监督场景

2. 预处理

    •    将 train 与 test 按行拼接得到整体数据集

    •    使用StandardScaler对所有特征做标准化（fit_transform 一次完成）

3. DBSCAN 聚类

    •    采用DBSCAN聚类，你可能会用到的参数固定为eps=0.3，min_samples=3, metric="euclidean", algorithm="auto"
4. 簇标签重映射（唯一化输出）

    •    设原本的标签集合为 {-1, 0, 1, …}，其中 -1 表示离群

    •    对所有非 -1 的簇 ℓ：

    i.    计算簇在标准化特征空间的质心 c_ℓ

    ii.    按质心第一维坐标从小到大排序得到顺序 ℓ₀, ℓ₁, …

    iii.    重新赋值：ℓ₀ → 0，ℓ₁ → 1，…

    •    离群点标签保持 -1 不变

5. 结果输出

    •    仅对 test 部分输出重新映射后的标签序列

    •    以单行 JSON 数组 输出


输入描述

标准输入仅一行 JSON，示例：

{

  "train": [[0, 0], [0.1, 0], [5, 5]],

  "test" : [[0.05, 0.05], [9, 0]]

}

    •    所有数值为整数 / 浮点数，无空行


输出描述

标准输出仅含一行：[0, -1]

    •    数组长度等于测试样本数

    •    逗号后须有空格，符合 JSON 规范


补充说明

1. 标准化：仅用一次 StandardScaler；不要对 train、test 分别拟合
2. 超参数：eps=0.3, min_samples=3 固定
3. 为了确保通过测试用例，仅允许使用numpy / pandas / scikit-learn

示例 1
收起 

输入
复制
{"train": [[0,0],[0.1,0],[5,5],[5.1,5],[10,0]], "test": [[0.05,0.05],[5.05,5.05],[9,0]]}
输出
复制
[0, 1, -1]
'''
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 读取输入数据
input_data = json.loads(input())
train_data = np.array(input_data["train"])
test_data = np.array(input_data["test"])

# 1. 拼接train与test数据
all_data = np.vstack((train_data, test_data))

# 2. 标准化处理
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data)

# 3. DBSCAN聚类
dbscan = DBSCAN(eps=0.3, min_samples=3, metric="euclidean", algorithm="auto") # 采用DBSCAN聚类，你可能会用到的参数固定为eps=0.3，min_samples=3, metric="euclidean",
labels = dbscan.fit_predict(all_data_scaled)

# 4. 簇标签重映射
# 获取所有非-1的簇标签
cluster_labels = [label for label in np.unique(labels) if label != -1]

# 计算每个簇的质心
cluster_centers = {}
for label in cluster_labels:
    cluster_points = all_data_scaled[labels == label]
    centroid = np.mean(cluster_points, axis=0)  # 计算质心
    cluster_centers[label] = centroid

# 按质心第一维坐标从小到大排序
sorted_labels = sorted(cluster_labels, key=lambda x: cluster_centers[x][0])

# 创建标签映射字典
label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
label_mapping[-1] = -1  # 离群点保持-1不变

# 重映射所有标签
remapped_labels = np.array([label_mapping[label] for label in labels])

# 5. 提取test部分的标签并输出
test_result = remapped_labels[len(train_data):].tolist()
print(json.dumps(test_result))

