import psutil
import os

# 获取当前进程的内存占用
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # 转换为 MB

if __name__ == "__main__":
    print(f"程序启动时的内存占用: {get_memory_usage()} MB")
    # 模拟一些内存操作
    data = [x ** 2 for x in range(10**6)]
    print(f"计算后内存占用: {get_memory_usage()} MB")
