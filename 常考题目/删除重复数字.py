from typing import List



class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        low_pointer = 0
        for i in range(1,len(nums)):
            if nums[i] != nums[low_pointer]:
                low_pointer += 1 # 这个必须在前
                nums[low_pointer] = nums[i]
            
        return low_pointer+1
        

# val_list = list(map(int,(input().split())))
solution = Solution()
# solution.removeDuplicates(val_list)
# print(val_list)

# 如果是无序。先排序：
def quickSort(lists, i ,j):
    # 递归结束条件
    if(i >= j):
        return lists
    
    pivot = lists[i]
    low, high = i, j
    while(i < j):
        while i < j and pivot <= lists[j]:
            j -= 1
        lists[i] = lists[j] # 中枢替换
        while i < j and pivot >= lists[i]:
            i += 1
        lists[j] = lists[i]
    lists[i] = pivot
    quickSort(lists, low, i-1)
    quickSort(lists, i+1, high)
    return lists

val_list = list(map(int,(input().split())))
val_list = quickSort(val_list, 0 ,len(val_list)-1)
len = solution.removeDuplicates(val_list)
print(len)
for i in range(0,len):
    print(val_list[i], end=' ')
    
    
def quickSort(lists, i ,j):
    # 递归结束条件
    if(i >= j):
        return lists
    
    pivot = lists[i]
    low, high = i, j
    while(i < j):
        while i < j and pivot <= lists[j]:
            j -= 1
        lists[i] = lists[j] # 中枢替换
        while i < j and pivot >= lists[i]:
            i += 1
        lists[j] = lists[i]
    lists[i] = pivot
    quickSort(lists, low, i-1)
    quickSort(lists, i+1, high)
    return lists
