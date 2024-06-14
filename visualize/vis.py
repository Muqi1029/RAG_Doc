import sys
sys.path.append("../")
from utils import read_json
import matplotlib.pyplot as plt



data = read_json("./result.json")
# 提取模型和指标
models = list(data.keys())
metrics = list(data["baseline"].keys())
recall_values = [data[model]["Recall_at_3"] for model in models]
mrr_values = [data[model]["MRR_at_3"] for model in models]

# 创建一个新的图形
plt.figure(figsize=(12, 8))

# 设置柱状图宽度
bar_width = 0.35

# 计算每个模型的 x 坐标位置
x = range(len(models))

# 绘制 Recall_at_3 的柱状图
plt.bar(x, recall_values, width=bar_width, color='blue', alpha=0.6, label='Recall at 3')

# 绘制 MRR_at_3 的柱状图，稍微调整 x 坐标位置以分开两个柱状图
plt.bar([i + bar_width for i in x], mrr_values, width=bar_width, color='green', alpha=0.6, label='MRR at 3')

# 设置 x 轴标签
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Comparison of Recall and MRR at 3 among Models')

# 设置 x 轴刻度和标签
plt.xticks([i + bar_width / 2 for i in x], models)
plt.xticks(rotation=45)

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.savefig("./result.png", dpi=800)
plt.show()