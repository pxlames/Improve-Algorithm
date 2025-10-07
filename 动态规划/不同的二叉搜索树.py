# 重叠子问题

# 通过dp[1] 和 dp[2] 来推导出来dp[3]的某种方式。
# dp[3] = dp[2] * dp[0] + dp[1] * dp[1] + dp[0] * dp[2]

# 1. dp[i] ： 1到i为节点组成的二叉搜索树的个数为dp[i]。
# 2. 在上面的分析中，其实已经看出其递推关系， dp[i] += dp[以i-j为头结点左子树节点数量] * dp[以j为头结点右子树节点数量]
# dp[i] += dp[j - 1] * dp[i - j]; (j从1开始)
# 3. 初始化dp[0]=1

class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[0] = 1
        for i in range(1,n+1):
            for j in range(1,i+1):
                dp[i] += dp[j-1] * dp[i-j]
        return dp[n]