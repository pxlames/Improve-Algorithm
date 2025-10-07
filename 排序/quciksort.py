def quickSort(lists, i, j):
    
    if(i >= j): 
        return lists
    
    pivot = lists[i]
    low, high = i, j
    while(i < j):
        while i < j and pivot <= lists[j]:
            j -= 1
        lists[i] = lists[j]
        while i < j and pivot >= lists[i]:
            i += 1
        lists[j] = lists[i]
    lists[i] = pivot # 注意i就是最中间那个中枢！
    quickSort(lists,low,i-1)
    quickSort(lists,i+1,high) # 注意这里是i
    return lists
            
if __name__ == '__main__':
    lists = [30, 24, 5, 58, 18, 36, 12, 42, 39]
    print('排序前的序列为：')
    for i in lists:
        print(i, end = " ")
        print('\n排列后的序列为：')
    for i in quickSort(lists, 0, len(lists)-1):
        print(i, end = " ")