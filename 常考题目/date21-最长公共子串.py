from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # 特殊情况
        if len(strs) == 0: return ""
        elif len(strs) == 1: return strs[0]
        elif len(strs) == 2: return self.findCommonPrefix(strs[0],strs[1])
        
        result = strs[0]
        for i in range(1,len(strs)):
            result = self.findCommonPrefix(result, strs[i])
        return result
        
    def findCommonPrefix(self, s1, s2):
        n = len(s1) if len(s1) < len(s2) else len(s2)
        result = ""
        for i in range(n):
            if s1[i] != s2[i]:
                break
            result += s1[i]
        return result