'''
示例 1：

输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
示例 2：

输入：s = "cbbd"
输出："bb"


'''
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if s == None:
            return ""
        size = len(s)
        left = 0
        right = 0
        maxLen = 0
        maxStart = 0
        
        for i in range(size):
            length = 1
            left = i - 1
            right = i + 1
            
            # 右边是否有对称
            while right <= size-1 and s[right] == s[i]:   ## 注意等号，因为是都要判断到！
                length += 1 # 找到一个相等的
                right += 1
            # 左边是否有对称
            while left >= 0 and s[left] == s[i]:
                length += 1
                left -= 1
            # 左右两边有一对相等的
            while left >= 0 and right <= size-1 and s[left] == s[right]: ## 注意+2
                length += 2
                left -= 1
                right += 1
            if length > maxLen:  ## 注意保存当前最大的left
                maxLen = length
                maxStart = left
        return s[maxStart+1:maxStart+maxLen+1] ## 注意+1，因为减过1；第二个参数是开，到这个索引-1