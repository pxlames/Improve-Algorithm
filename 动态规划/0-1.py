

from typing import List


def zeo_one_knapsack(capacity: int, w: List[int], v: List[int]) -> int:
    n = len(w)

    # 从最后一个i开始思考，选和不选两种状态，不选的话如何，选的话如何。求最大max。
    def dfs(i, c):
        if i < 0:
            return 0 # 回溯终止条件
        if c < w[i]: # 边界条件
            return dfs(i-1, c)
        return max(dfs(i-1,c), dfs(i-1, c-w[i])) + v[i]

def unbounded_knapsack(capacity: int, w: List[int], v: List[int]) -> int:
    n = len(w)

    # 从最后一个i开始思考，选和不选两种状态，不选的话如何，选的话如何。求最大max。
    def dfs(i, c):
        if i < 0:
            return 0 # 回溯终止条件
        if c < w[i]: # 边界条件
            return dfs(i-1, c)
        return max(dfs(i-1,c), dfs(i, c-w[i])) + v[i]

from typing import List
def zeo_one_knapsack_dp(n: int, bagweight: int, weight: List[int], value: List[int]):
    dp = [[0] * (bagweight + 1) for _ in range(n)]
    for j in range(weight[0], bagweight + 1):
        dp[0][j] = value[0]
    for i in range(1, n):
        for j in range(bagweight + 1):
            if j < weight[i]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i] + value[i]])
    print(dp[n-1][bagweight])
    

# 完全背包：
def unbounded_knapsack(n, bag_weight, weight, value):
    '''
    n是代表几种物品
    '''
    dp = [[0] * (bag_weight + 1) for _ in range(n)]

    # 初始化，对只有一个物品时的情况全部设置值
    for j in range(weight[0], bag_weight + 1):
        dp[0][j] = dp[0][j - weight[0]] + value[0]

    # 动态规划
    for i in range(1, n):
        for j in range(bag_weight + 1):
            if j < weight[i]:# 不能选
                dp[i][j] = dp[i - 1][j]
            else:# 能选，这里区别于0-1背包的点：i不用-1
                dp[i][j] = max(dp[i - 1][j], dp[i][j - weight[i]] + value[i])

    return dp[n - 1][bag_weight]