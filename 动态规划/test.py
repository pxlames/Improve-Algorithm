

def game(n,a,b):
    dp = [[0,0] for _ in range(n)]
    # 初始状态
    dp[0][0] = a[0]
    dp[0][1] = 0

    for i in range(1,n):
        # 定义转移方程：
        dp[i][0] = max(dp[i-1][0] + a[i],dp[i-1][1] + b[i-1])
        dp[i][1] = dp[i-1][0]
        

    result = max(dp[n-1][0],dp[n-1][1])
    return result

n = int(input())
a = list(map(int,input().split()))
b = list(map(int,input().split()))

result = game(n,a,b)
print(result)