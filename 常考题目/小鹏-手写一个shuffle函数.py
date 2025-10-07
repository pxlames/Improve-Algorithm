import random

def my_shuffle(lst):
    """
    手动实现洗牌功能，打乱列表元素顺序
    
    参数:
        lst: 需要打乱的列表
        
    返回:
        打乱后的新列表（不修改原列表）
    """
    # 创建列表副本，避免修改原列表
    result = lst.copy()
    # 获取列表长度
    length = len(result)
    
    # Fisher-Yates洗牌算法
    # 从最后一个元素开始，逐个向前遍历
    for i in range(length - 1, 0, -1):
        # 生成一个0到i之间的随机整数（包括0和i）
        j = random.randint(0, i)
        # 交换位置i和位置j的元素
        result[i], result[j] = result[j], result[i]
    
    return result

# 测试示例
if __name__ == "__main__":
    original = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    shuffled = my_shuffle(original)
    
    print("原始列表:", original)
    print("打乱后列表:", shuffled)
