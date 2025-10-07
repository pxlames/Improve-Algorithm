from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        size = len(nums)
        for i in range(size):
            for j in range(i+1,size,1):
                if nums[i] + nums[j] == target:
                    return [i,j]

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}
        for i, num in enumerate(nums):
            map[num] = i
        # O（n）遍历
        for i, num in enumerate(nums):
            find = map.get(target-num, -1)
            if find != -1 and find != i: return [i,find]