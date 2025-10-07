'''

小美有 n 个长方形，第 i 个长方形的两条边长分别为 x_i,y_i；
\hspace{15pt}小美拥有一个仅包含第一象限的平面直角坐标系；
\hspace{15pt}小美希望将这 n 个长方形按顺序（可以旋转）放置在 x 轴上，不允许重叠，并且每个长方形放置后的高度不超过 m，保证\max(\min(x_i,y_i)) \leqq m；

\hspace{15pt}请问，在满足上述条件的前提下，小美最少需要占用 x 轴的长度是多少？

输入描述
\hspace{15pt}第一行输入两个整数 n,m\ \left(1\leqq n\leqq2\times10^5;\ 1\leqq m\leqq10^9\right)，分别表示长方形的个数和允许的最大高度；
\hspace{15pt}接下来 n 行，每行输入两个整数 x_i,y_i\ \left(1\leqq x_i,y_i\leqq10^9\right)，分别表示第 i 个长方形的两条边长。

输出描述
\hspace{15pt}输出一个整数，表示在每个长方形高度不超过 m 的情况下，所需占用的最短 x 轴长度。

输入:
3 4
1 2
2 5
4 2
输出:
8
说明
\hspace{23pt}\bullet\,第 1 个长方形可用高度较小的一条边放置，高度 2\leqq4，占用长度 \min(1,2)=1；
\hspace{23pt}\bullet\,第 2 个长方形仅能以高度 2 放置，占用长度 5；
\hspace{23pt}\bullet\,第 3 个长方形可用较小的一条边放置，占用长度 \min(4,2)=2；
\hspace{15pt}因此总长度为 1+5+2=8。
'''

import sys

for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))


# 最终计算总长度返回
n,m = map(int,input().split())
total = 0
for _ in range(n):
    x,y = map(int,input().split())
    if x <= m and y <= n:
        total += max(x,y)
    else:
        total += x + y - min(x,y)
        
print(total)

