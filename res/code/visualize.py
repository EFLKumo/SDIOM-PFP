import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

station_name = input("输入站点名：")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 ['Microsoft YaHei']  选择一个你电脑上有的字体
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号，防止负号显示为方块

# 1. 选择文件
file_path = 'output_stations/' + station_name + ".txt"

# 2. 读取文件数据
try:
    with open(file_path, 'r') as f:
        time_strings = f.read().strip().split(',')
except FileNotFoundError:
    print(f"文件 {file_path} 未找到。")
    exit()

# 3. 转换时间字符串为 datetime 对象
try:
    time_series = pd.to_datetime(time_strings, format='%Y-%m-%d %H:%M:%S')
except ValueError as e:
    print(f"时间字符串解析错误，检查格式\n错误信息: {e}")
    exit()

# 4. 创建 Pandas Series
time_series_pd = pd.Series(time_series)

# 5. 数据分箱和聚合 (按分钟)
df = pd.DataFrame({'time': time_series_pd})
df['time_bin'] = df['time'].dt.floor('T')
hourly_counts = df.groupby('time_bin').size().reset_index(name='count')
hourly_counts = hourly_counts.set_index('time_bin')

# 正确计算分钟数的方法：
if not hourly_counts.index.empty: # 检查 index 是否为空，避免报错
    first_date = hourly_counts.index[0].date() # 获取第一个时间戳的日期
    start_of_day = pd.to_datetime(first_date) # 使用该日期创建 start_of_day
    hourly_counts['minutes'] = (hourly_counts.index - start_of_day).total_seconds() / 60
else:
    hourly_counts['minutes'] = 0  # 如果 index 为空，minutes 设为 0

# 6. 应用平滑技术
hourly_counts['sma'] = hourly_counts['count'].rolling(window=25, center=True, min_periods=1).mean()

smoothed_loess = lowess(hourly_counts['count'], hourly_counts['minutes'], frac=0.1)
hourly_counts['loess'] = smoothed_loess[:, 1]

smoothed_loess_15 = lowess(hourly_counts['count'], hourly_counts['minutes'], frac=0.15)
hourly_counts['loess_0.15'] = smoothed_loess_15[:, 1]

smoothed_loess_2 = lowess(hourly_counts['count'], hourly_counts['minutes'], frac=0.2)
hourly_counts['loess_0.2'] = smoothed_loess_2[:, 1]

# 7. 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(hourly_counts['minutes'], hourly_counts['count'], label='原始数据', alpha=0.5)
plt.plot(hourly_counts['minutes'], hourly_counts['sma'], label='SMA (25 分钟窗口)', linewidth=2)
plt.plot(hourly_counts['minutes'], hourly_counts['loess'], label='LOESS (frac=0.1)', linewidth=2)
plt.plot(hourly_counts['minutes'], hourly_counts['loess_0.15'], label='LOESS (frac=0.15)', linewidth=2)
plt.plot(hourly_counts['minutes'], hourly_counts['loess_0.2'], label='LOESS (frac=0.2)', linewidth=2)

plt.xlabel('时间 (自 0:00 的分钟数)')
plt.ylabel('入站人数')
plt.title('入站人数曲线 (' + station_name + '站)')
plt.xlim(0, 48 * 60)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
