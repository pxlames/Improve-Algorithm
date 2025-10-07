class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def backtrack(first=0):
            # 所有数都填完了
            if first == n:
                res.append(nums[:])
            for i in range(first, n):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]

        n = len(nums)
        res = []
        backtrack()
        return res


solution = Solution()
totalAmount = 500.00
invoice = [1, 2, 34, 52, 43.33, 44.33]
res = solution.permute(invoice)
maxIndex = 0
maxValue = 0
for index,arr in enumerate(res):
    if(sum(arr) > maxValue):
        maxIndex = index
        maxValue = sum(arr)
print(maxIndex)
print(maxValue)
print(res[maxIndex])



