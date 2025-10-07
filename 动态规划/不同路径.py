class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        k = max(m,n)
        # dp[i][j] ：表示从（0 ，0）出发，到(i, j) 有dp[i][j]条不同的路径。
        dp = [[0] * n for _ in range(m)]
        # 初始化这里想错了，向右向下就只有一条路
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        # 确定便利顺序：这里要看一下递推公式dp[i][j] = dp[i - 1][j] + dp[i][j - 1]，dp[i][j]都是从其上方和左方推导而来，那么从左到右一层一层遍历就可以了。
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j]= dp[i][j-1] + dp[i-1][j]
        return dp[m-1][n-1]