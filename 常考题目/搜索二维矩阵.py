class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        start = 0
        end = len(matrix[0]) * len(matrix) - 1
        while start <= end: # 注意小于等于，确保最后一个也被判断到
            mid = (start + end) // 2
            mid_val = matrix[mid // len(matrix[0])][mid % len(matrix[0])]
            if mid_val == target: # 找到了
                return True
            elif mid_val > target:
                end = mid - 1
            else:
                start = mid + 1

        return False