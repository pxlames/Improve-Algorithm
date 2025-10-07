
# 回溯+cache=记忆话搜索=>递推版本
nums = input()
val = list(map(int,nums.strip('[]').split(','))) # 学习到了如何读取非acm的输入
print(val)

res_0 = res_1 = 0
for i,x in enumerate(val):
    res_new = max(res_1, res_0 + x)
    res_0 = res_1
    res_1 = res_new
print(res_1)