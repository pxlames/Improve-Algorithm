from typing import List


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def reverse(lists,i,j):
            while i < j:
                lists[i],lists[j] = lists[j],lists[i]
                i += 1
                j -= 1
        # 特殊情况
        if nums is None: return
        if len(nums) == 0: return 
        size = len(nums)
        k = k % size #### # 轮转 k 次等于轮转 k % n 次!
        reverse(nums,0,size-1)
        reverse(nums,0,k-1)
        reverse(nums,k,size-1)