from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        size = len(nums)
        sorted_nums = self.quickSort(nums, 0, size-1)
        for i in range(0,size-1,2):
            if sorted_nums[i] != sorted_nums[i+1]:
                return sorted_nums[i]
        return sorted_nums[size-1] # 最后一个

    def quickSort(self, list, i, j):
        if(i >= j):
            return list
        left = i
        right = j
        while i < j:
            # 找到一个不满足的j
            pivot = list[i]
            while list[j] >= pivot and i < j:
                j -= 1
            list[i] = list[j]
            while list[i] <= pivot and i < j:
                i += 1
            list[j] = list[i]
            list[i] = pivot
        self.quickSort(list, left, i-1)
        self.quickSort(list, i+1, right)
        
        return list



from typing import List

class Solution2:
    def singleNumber(self, nums: List[int]) -> int:
        sets = set()
        for num in nums:
            if num not in sets:
                sets.add(num)
            else:
                sets.remove(num)
        return sets.pop()
    
    
from typing import List

class Solution3:
    def singleNumber(self, nums: List[int]) -> int:
        map = {}
        for num in nums:
            map[num] = map.get(num,0) + 1
        for num in nums:
            if map[num] == 1:
                return num
            
class Solution4:
    def singleNumber(self, nums: List[int]) -> int:
        sets = set(nums)
        result = 3 * sum(sets) - sum(nums)
        return result / 2

s = Solution()
# input_list = list(map(int, input().split()))
input_list = [-336,513,-560,-481,-174,101,-997,40,-527,-784,-283,-336,513,-560,-481,-174,101,-997,40,-527,-784,-283,354]
# print(input_list)
result = s.singleNumber(input_list)
print(result)

