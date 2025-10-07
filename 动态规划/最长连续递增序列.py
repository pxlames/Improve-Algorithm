# dp[i]表示i之前包括i的以nums[i]结尾的最长递增子序列的长度

# if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1);

# 每一个i，对应的dp[i]（即最长递增子序列）起始大小至少都是1.
from typing import List
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        dp = [1] * len(nums)
        result = 1
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            result = max(result, dp[i]) #取长的子序列
        return result