# 1.dp[i]：爬到有i个台阶的楼顶，有dp[i]种方法。
# 2.递推公式为：dp[i] += dp[i - j]
# 3.dp[0] 一定为1
# 4.
# 5.

def climbing_stairs(n,m):
    dp = [0]*(n+1) # 背包总容量
    dp[0] = 1 
    # 排列题，注意循环顺序，背包在外物品在内
    for j in range(1,n+1): # 多少个
        for i in range(1,m+1): # 重量
            if j>=i:
                dp[j] += dp[j-i] # 这里i就是重量而非index
    return dp[n]