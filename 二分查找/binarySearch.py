from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while(left <= right):
            mid = left + (right - left) / 2
            
            if nums[mid] > target:
                right = mid - 1 # target在左区间，所以是[left, mid-1]
            elif nums[mid] < target:
                left = mid + 1 # target 在右区间，所以是[mid+1, right]
            else:
                return mid
        return -1