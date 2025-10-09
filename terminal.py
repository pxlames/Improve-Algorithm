import os

def set_proxy_env():
    """
    设置代理环境变量
    """
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

set_proxy_env()
print("代理环境变量已设置。")
print(os.environ["https_proxy"])
print(os.environ["http_proxy"])
print(os.environ["all_proxy"])



