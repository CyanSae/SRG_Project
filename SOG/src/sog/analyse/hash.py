from eth_utils import keccak

sig = 'exploit()'  # 尝试匹配你怀疑的函数
selector = keccak(text=sig)[:4].hex()
print("0x" + selector)  # 会输出选择器
