import pandas as pd
import os

# 1. 看看下了哪几个文件
print(os.listdir("./data/assist2009_hf/data"))

# 2. 读取这个唯一的文件
df = pd.read_parquet("./data/assist2009_hf/data/train-00000-of-00001.parquet")

# 3. 打印前 2 行看看长啥样
print(df.head(2))
# 还可以看看它有哪些列名
print(df.columns)