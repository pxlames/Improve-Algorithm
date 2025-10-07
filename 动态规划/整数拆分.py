
# dp[i]：分拆数字i，可以得到的最大乘积为dp[i]。

'''
其实可以从1遍历j，然后有两种渠道得到dp[i].
一个是j * (i - j) 直接相乘。
一个是j * dp[i - j]，相当于是拆分(i - j)，对这个拆分不理解的话，可以回想dp数组的定义。
'''
# dp[i] = max({dp[i], (i - j) * j, dp[i - j] * j});

# 3.dp[0] = 1，dp[1] = 1，dp[2] = 1 


class Solution:
    def integerBreak(self, n: int ) -> int:
        dp = [0] * (n+1)
        dp[2] = 1
        
        for i in range(3, n+1):
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[i], max(j * dp[i-j],j * (i-j)))
        return dp[n]