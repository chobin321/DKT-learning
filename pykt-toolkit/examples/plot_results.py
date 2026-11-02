import re
import matplotlib.pyplot as plt
import os

# 1. 读取刚才生成的日志文件（注意路径加了 logs/）
log_file = "logs/train_log.txt"
if not os.path.exists(log_file):
    print(f"❌ 找不到 {log_file}，请先运行带日志重定向的训练命令！")
    exit()

epochs = []
losses = []
aucs = []

# 2. 用正则表达式提取数据
with open(log_file, "r", encoding="utf-16") as f:
    for line in f:
        # 匹配形如 "Epoch: 19, validauc: 0.8955, ... train loss: 0.31..."
        match = re.search(r"Epoch:\s*(\d+).*?valid\d*auc:\s*([\d.]+).*?train[_\s]*loss:\s*([\d.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            aucs.append(float(match.group(2)))
            losses.append(float(match.group(3)))

# 3. 检查是否抓取到了数据
if not epochs:
    print("❌ 没能在日志里找到符合格式的数据，请检查日志文件内容！")
    exit()

print(f"✅ 成功提取了 {len(epochs)} 个 Epoch 的数据，正在绘制...")

# 4. 开始画图
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号
plt.figure(figsize=(12, 5))

# --- 左图：Loss 曲线 ---
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label='Train Loss', color='#1f77b4', linewidth=2)
plt.title('训练损失 (Train Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# --- 右图：AUC 曲线 ---
plt.subplot(1, 2, 2)
plt.plot(epochs, aucs, label='Valid AUC', color='#ff7f0e', linewidth=2)
plt.title('验证集 AUC (Validation AUC)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 调整布局并保存（注意路径也加了 logs/）
plt.tight_layout()
plt.savefig('logs/training_curves.png', dpi=300) 
print("🎉 图片已成功保存为 logs/training_curves.png")