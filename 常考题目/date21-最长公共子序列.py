'''
相对顺序是一致的。
但是这题需要用dp！


'''

class Solution:
    # 递归搜索 + 保存计算结果 = 记忆化搜索
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n,m = len(text1),len(text2)
        @cache
        def dfs(i,j):
            if i < 0 or j < 0: return 0
            if text1[i] == text2[j]:
                return 1 + dfs(i-1,j-1)
            return max(dfs(i,j-1),dfs(i-1,j)) # 只有这两种情况
        return dfs(n-1,m-1)
    # 递推
    def longestCommonSubsequence_2(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        for i, x in enumerate(s):
            for j, y in enumerate(t):
                f[i + 1][j + 1] = f[i][j] + 1 if x == y else \
                                  max(f[i][j + 1], f[i + 1][j])
        return f[n][m]
