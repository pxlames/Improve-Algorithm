from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = list()
        res = list()
        while root or stack: # 可以继续找
            while root: # 当前节点
                stack.append(root) # 入栈
                root = root.left # 左边全部入栈
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
# 先序遍历
class Solution2:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = list()
        res = list()
        while root or stack: # 可以继续找
            while root: # 当前节点
                stack.append(root) # 入栈
                res.append(root.val) # 先序遍历
                root = root.left # 左边全部入栈
            root = stack.pop()
            root = root.right
        return res

# 后序遍历
class Solution3:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = list()
        res = list()
        while root or stack: # 可以继续找
            while root: # 当前节点
                stack.append(root) # 入栈
                res.append(root.val) # 后序遍历
                root = root.left # 左边全部入栈
            root = stack.pop()
            root = root.right
        res.reverse()
        return res