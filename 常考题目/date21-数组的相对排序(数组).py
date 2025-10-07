'''
示例 1：

输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
输出：[2,2,2,1,4,3,3,9,6,7,19]
示例  2:

输入：arr1 = [28,6,22,8,44,17], arr2 = [22,28,8,6]
输出：[22,28,8,6,17,44]

'''


from typing import List

class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        maps = {}
        for item in arr1:
            maps[item] = maps.get(item,0) + 1
        print(maps)
        
        # 集合交集
        set1 = set(arr1)
        set2 = set(arr2)
        list3 = list(set1 - set2)
        list3.sort()
        print(list3)
        
        # 开始构建
        res = []
        for item in arr2+list3:
            for _ in range(maps[item]):
                res.append(item)

        return res
s = Solution()
arr1 = [2,3,1,3,2,4,6,7,9,2,19]
arr2 = [2,1,4,3,9,6]
s.relativeSortArray(arr1,arr2)