# Definition for singly-linked list.

'''
链表的都是多个指针！都不难！
'''
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode: