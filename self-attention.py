import numpy as np


def self_attention(q,k,v,mask=None):
    batch_size= q.shape[0]
    seq_len = q.shape[1]
    d_k = q.shape[2]
    
    scores = np.matmul(q,k,np.transpose(0,2,1)) # 计算注意力分数
    scores = scores / np.sqrt(d_k) # 缩放，防止太大
    if mask:
        scores = scores * (mask * -1e9)
    # 归一化得到注意力权重
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    output = np.matmul(attention_weights, v)
    return output

batch_size = 2
seq_len = 4
d_k = 64
d_v = 64

q = np.random.randn(batch_size,seq_len,d_k)
k = np.random.randn(batch_size,seq_len,d_k)
v = np.random.randn(batch_size,seq_len,d_v)

mask = np.zeros(batch_size,seq_len,seq_len)
mask[:,2:,:] = -1

print(self_attention(q,k,v,mask))

