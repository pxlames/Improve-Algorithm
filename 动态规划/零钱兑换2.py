'''
应该会知道类似这种题目：给出一个总数，一些物品，问能否凑成这个总数。

这是典型的背包问题！
本题求的是装满这个背包的物品组合数是多少。



'''
from typing import List
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0]*(amount + 1)
        dp[0] = 1
        # 遍历物品
        for i in range(len(coins)):
            # 遍历背包
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
        return dp[amount]