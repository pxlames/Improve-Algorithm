if __name__ == '__main__':
    sum = 0
    for i in range(1,2020):
        s = str(i)
        if '0' in s or '1' in s or '2' in s or '9' in s:
            sum += i*i
    print(sum)