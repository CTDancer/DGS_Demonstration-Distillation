import matplotlib.pyplot as plt
import numpy as np

# 原始数据列表
data = [330, 480, 1060, 515, 172, 394, 383, 557, 556, 426, 566, 503, 636, 627, 401, 902, 498, 382, 422, 587, 731, 656, 330, 424, 465, 404, 337, 569, 739, 328, 592, 557, 811, 658, 516, 586, 380, 404, 890, 531, 494, 373, 585, 530, 518, 388, 699, 599, 537, 655]

# 设置直方图的区间
bins = np.arange(0, 1060, 100)

# 统计每个区间内的数据点数量
hist, edges = np.histogram(data, bins=bins)

# 绘制直方图
plt.hist(data, bins=bins, edgecolor='k', alpha=0.7)
plt.xlabel('# tokens of distilled prompts')
plt.ylabel('count')

# 显示图形
plt.show()