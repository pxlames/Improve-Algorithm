class Solution:
    def strangePrinter(self, s: str) -> int:
        # 五部曲
        # 1.dp[i][j],在区间[i,j]需要最少的打印次数
        # 2.递推公式，左右相等，dp[i][j] = dp[i][j-1];否则枚举两部分 f[i][k] + f[k+1][j]
        # 3.初始值、边界条件dp[i][i] = 1 最终返回 dp[0][n-1]

        n = len(s)
        dp = [[0] * n for _ in range(n+1)]
        for l in range(n): # 区间长度
            for i in range(n-l): # 区间起点
                j = i + l # 区间终点
                dp[i][j] = dp[i+1][j] + 1
                for k in range(i+1, j+1):
                    if s[k] == s[i]:
                        dp[i][j] = min(dp[i][j], dp[i][k-1]+dp[k+1][j])
        return dp[0][-1]
