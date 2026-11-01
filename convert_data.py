import pandas as pd
import numpy as np
import os

print("正在读取 parquet 文件...")
df = pd.read_parquet("./data/assist2009_hf/data/train-00000-of-00001.parquet")

# 1. 为了安全，把所有的 skill_ids 映射成从 0 开始的纯数字
# (因为之前的数据里可能夹杂着 'algebra' 这样的字母，pyKT 要求必须是纯数字)
all_skills = set()
for skills in df['skill_ids']:
    all_skills.update(skills)
skill2idx = {s: i for i, s in enumerate(all_skills)}

print(f"共有 {len(skill2idx)} 个不同的知识点")

# 2. 构造 pyKT 需要的序列格式
MAX_LEN = 200
sequences = []
for _, row in df.iterrows():
    user_id = row['user_id']
    skill_ids = row['skill_ids']
    grades = row['grades']
    
    if len(skill_ids) < 3:
        continue
        
    # 提取序列并截断到 200
    q_list = [skill2idx[s] for s in skill_ids][:MAX_LEN]
    r_list = [int(g) for g in grades][:MAX_LEN]
    
    real_len = len(q_list)
    
    # 如果长度小于 200，补齐 0
    q_list = q_list + [0] * (MAX_LEN - real_len)
    r_list = r_list + [0] * (MAX_LEN - real_len)
    
    # selectmasks：真实题目位置填 1，补零位置填 0
    mask_list = [1] * real_len + [0] * (MAX_LEN - real_len)
    
    # 转成逗号分隔的字符串
    q_str = ",".join([str(s) for s in q_list])
    c_str = q_str  
    r_str = ",".join([str(s) for s in r_list])
    masks_str = ",".join([str(s) for s in mask_list])
    
    sequences.append({
        'user_id': user_id,
        'questions': q_str,
        'concepts': c_str,
        'responses': r_str,
        'selectmasks': masks_str,
        'fold': 0
    })

seq_df = pd.DataFrame(sequences)
print("转换完成，共提取了", len(seq_df), "个有效学生序列")

# 3. 划分训练集和测试集（80%训练，20%测试）
train_df = seq_df.sample(frac=0.8, random_state=42)
test_df = seq_df.drop(train_df.index)

# 🌟 新增：给训练集随机分配 5 折（0, 1, 2, 3, 4）
np.random.seed(42)  # 保证每次生成的随机数一样
train_df['fold'] = np.random.randint(0, 5, size=len(train_df))

# 测试集随便给个 0 就行（因为测试集读取时不受这个影响）
test_df['fold'] = -1

# 4. 保存
os.makedirs("./pykt-toolkit/data/assist2009", exist_ok=True)
train_df.to_csv("./pykt-toolkit/data/assist2009/train_valid_sequences.csv", index=False)
test_df.to_csv("./pykt-toolkit/data/assist2009/test_sequences.csv", index=False)

print("成功！数据已保存到 ./pykt-toolkit/data/assist2009/ 目录下")