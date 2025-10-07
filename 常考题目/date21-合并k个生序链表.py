# Definition for singly-linked list.

from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # 顺序合并
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        return None
    
    # 分治归并
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]):
        cur = res = ListNode()
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        cur.next = list1 if list1 else list2
        return res.next
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 特殊情况
        m = len(lists)
        if m == 0:
            return None
        if m == 1:
            return lists[0]
        # 归并处理
        left = mergeKLists(lists[:m//2])
        right = mergeKLists(lists[m//2:])
        return self.mergeTwoLists(left, right)
    
    # 优先级队列（大概先作了解）PriorityQueue
    