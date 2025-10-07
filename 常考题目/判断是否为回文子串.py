class Solution:
    def countSubstrings(self, s: str) -> int:
        result = 0
        for i in range(len(s)):
            result += self.sub_cnt(s,i,i)# 回文字母数为奇数
            result += self.sub_cnt(s,i,i+1)# 回文字母数为偶数
        return result

    def sub_cnt(self, s, start, end):
        result = 0
        while start >= 0 and end < len(s):
            if s[start] != s[end]:# 不是回文
                break
            else:
                result += 1 # 是回文，左右相等即是
                start -= 1 # 往左
                end += 1 # 往右
        return result 

'''
采用了中心扩展法的思路
回文子串的特点是 "正着读和反着读完全相同"（如 "aba"、"aa"）。
'''