# 写一个执行python的函数，输入python代码，返回执行结果

def execute_python(code):
    """
    执行python代码，返回执行结果
    """
    return exec(code) # 解释为啥可以执行python代码
    # exec 函数可以执行动态创建的 Python 代码字符串。它与 eval 类似，但更安全，因为它不返回结果，而是直接执行代码。
    # 当传入的 code 参数是一个有效的 Python 代码字符串时，exec 会执行该代码。
    # 这里我们传入了一个简单的 print 语句，exec 会执行这个语句并打印 "Hello, World!"。
    # 因此，execute_python 函数会返回 "Hello, World!"。
    # 注意：在生产环境中，应该避免使用 exec 来执行不可信的代码，以防止安全风险。
    # 推荐使用更安全的替代方法，例如使用 ast.literal_eval 来安全地执行代码。

print(execute_python("print('Hello, World!')")) # Hello, World
print(execute_python("print(1+1)")) # 2
