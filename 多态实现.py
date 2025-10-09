'''
真实生产环境中的多态示例：
以支付系统为例，不同的支付方式（如支付宝、微信、银行卡）都实现了统一的支付接口，业务代码只依赖接口，不关心具体实现。
'''

from abc import ABC, abstractmethod

# 定义支付接口
class Payment(ABC):
    @abstractmethod
    def pay(self, amount: float) -> None:
        pass

# 支付宝支付实现
class AliPay(Payment):
    def pay(self, amount: float) -> None:
        print(f"使用支付宝支付{amount}元")

# 微信支付实现
class WeChatPay(Payment):
    def pay(self, amount: float) -> None:
        print(f"使用微信支付{amount}元")

# 银行卡支付实现
class BankCardPay(Payment):
    def pay(self, amount: float) -> None:
        print(f"使用银行卡支付{amount}元")

# 订单结算，依赖于支付接口
class Order:
    def __init__(self, payment: Payment):
        self.payment = payment

    def checkout(self, amount: float):
        # 这里体现多态：实际调用的是传入对象的pay方法
        self.payment.pay(amount)

# 客户端代码
if __name__ == "__main__":
    # 用户动态选择支付方式
    payment_methods = [AliPay(), WeChatPay(), BankCardPay()]
    order = Order(None)  # 先不指定支付方式

    # 假设有一组支付方式和金额
    payments = [
        (AliPay(), 100.0),
        (WeChatPay(), 200.0),
        (BankCardPay(), 300.0)
    ]

    # 只用一个order对象，动态切换支付方式并结算
    order.payment, amount = payments[0]
    order.checkout(amount)

    order.payment, amount = payments[1]
    order.checkout(amount)

    order.payment, amount = payments[2]
    order.checkout(amount)