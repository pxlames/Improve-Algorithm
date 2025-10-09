from typing import List
class Solution:
    def trap(self, height: List[int]) -> int:
        # 单调栈：满足从栈底到栈顶的下标对应的数组 height 中的元素递减。
        stack = list()
        ans = 0 
        n = len(height)

        for i,h in enumerate(height):
            while stack and h > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                left = stack[-1]
                currWidth = i - left - 1
                currHeight = min(height[left],height[i]) - height[top]
                ans += currWidth * currHeight
            stack.append(i)
        return ans

# 学会栈的使用！
'''
list()

list.append(i) 入栈
list.pop() 出栈
list[-1] 栈顶元素
'''