
# 必须要有序
def binarySearch(x):
    i = 0
    j = x-1
    while i <= j:
        mid = (i + j) // 2
        if mid*mid == x: return mid
        elif mid*mid < x: 
            i = mid + 1
        else:
            j = mid - 1
    return i-1
x = int(input())
print(binarySearch(x))

def binarySearch(x):
    i = 0
    j = x-1
    while i <= j:
        mid = (i+j) // 2
        if mid * mid == x: return mid
        elif mid * mid > x:
            j = mid - 1
        else:
            i = mid + 1
    return i-1