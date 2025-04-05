import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
REFERENCE_STATION = "五和" # 参考站点名称
TEST_STATION = input("站点名：") # 测试站点名称
DATA_DIR = "output_stations" # 数据文件目录
LOESS_FRAC = 0.2 # LOESS 平滑参数，控制平滑度

def min_max_normalize(data):
    """
    对数据进行 Min-Max 归一化。

    参数:
    data (np.array): 要归一化的数据数组。

    返回:
    np.array: 归一化后的数据数组。
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data) # 避免除零错误，返回全零数组
    return (data - min_val) / (max_val - min_val)

def load_and_process(station_name):
    """
    加载指定站点的数据，进行时间序列处理、分箱聚合和 LOESS 平滑。

    参数:
    station_name (str): 站点名称。

    返回:
    tuple: 包含原始计数数据 (DataFrame) 和平滑计数数据 (DataFrame)，
           如果加载或处理失败，则返回 (None, None)。
    """
    file_path = os.path.join(DATA_DIR, f"{station_name}.txt")
    try:
        with open(file_path, 'r') as f:
            time_strings = f.read().strip().split(',')
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return None, None

    try:
        time_series = pd.to_datetime(time_strings, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except ValueError as e:
        print(f"错误: 时间字符串解析错误，请检查日期时间格式\n详细错误: {e}")
        return None, None

    time_series_pd = pd.Series(time_series)
    hourly_counts = pd.DataFrame({'time': time_series_pd}).groupby(pd.Grouper(key='time', freq='T')).size().reset_index(name='count')
    hourly_counts = hourly_counts.rename(columns={'time': 'time_bin'}).set_index('time_bin')

    if not hourly_counts.index.empty:
        start_of_day = hourly_counts.index[0].normalize() # 获取当天的开始时间
        hourly_counts['minutes'] = (hourly_counts.index - start_of_day).total_seconds() / 60
    else:
        hourly_counts['minutes'] = 0

    raw_counts = hourly_counts[['minutes', 'count']].copy()

    smoothed_data = lowess(hourly_counts['count'], hourly_counts['minutes'], frac=LOESS_FRAC)
    hourly_counts['smooth'] = smoothed_data[:, 1]
    hourly_counts['smooth_normalized'] = min_max_normalize(hourly_counts['smooth'])

    return raw_counts, hourly_counts

# 加载数据
ref_raw_data, ref_processed_data = load_and_process(REFERENCE_STATION)
if ref_raw_data is None:
    exit()

test_raw_data, test_processed_data = load_and_process(TEST_STATION)
if test_raw_data is None:
    exit()

# --- 可视化原始数据和平滑结果 ---
plt.figure(figsize=(12, 6))
plt.plot(ref_raw_data['minutes'], ref_raw_data['count'], label=f'{REFERENCE_STATION}站原始数据', alpha=0.5)
plt.plot(ref_processed_data['minutes'], ref_processed_data['smooth'], label=f'{REFERENCE_STATION}站 LOESS (frac={LOESS_FRAC})', linewidth=2)
plt.plot(test_raw_data['minutes'], test_raw_data['count'], label=f'{TEST_STATION}站原始数据', alpha=0.5)
plt.plot(test_processed_data['minutes'], test_processed_data['smooth'], label=f'{TEST_STATION}站 LOESS (frac={LOESS_FRAC})', linewidth=2)

plt.xlabel('时间 (分钟)')
plt.ylabel('入站人数')
plt.title(f'{TEST_STATION}站与{REFERENCE_STATION}站入站人数曲线')
plt.xlim(0, 1440) # 24小时分钟数
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --- DTW 对齐 ---
# 使用归一化平滑数据进行 DTW 对齐
distance, path = fastdtw(ref_processed_data['smooth_normalized'].values.reshape(-1, 1),
                       test_processed_data['smooth_normalized'].values.reshape(-1, 1),
                       dist=euclidean)

# 初始化对齐序列和计数器
aligned_test_smooth = np.zeros_like(ref_processed_data['smooth'].values)
alignment_counts = np.zeros_like(ref_processed_data['smooth'].values, dtype=float)

# 基于 DTW 路径对齐测试站点的平滑数据
for i, j in path:
    if i < len(aligned_test_smooth) and j < len(test_processed_data['smooth'].values):
        aligned_test_smooth[i] += test_processed_data['smooth'].values[j]
        alignment_counts[i] += 1

# 计算平均对齐数据，避免除以零
aligned_test_smooth = np.divide(aligned_test_smooth, alignment_counts, out=np.zeros_like(aligned_test_smooth), where=alignment_counts != 0)
normalized_aligned_smooth = min_max_normalize(aligned_test_smooth) # 归一化对齐后的数据

# --- 可视化 DTW 对齐效果 ---
plt.figure(figsize=(12, 6))
plt.plot(ref_processed_data['minutes'], ref_processed_data['smooth'], label=f'{REFERENCE_STATION}站 LOESS', linewidth=2)
plt.plot(test_processed_data['minutes'], test_processed_data['smooth'], label=f'{TEST_STATION}站 LOESS', linewidth=2)
plt.plot(ref_processed_data['minutes'], aligned_test_smooth, label='DTW 对齐后数据', linewidth=2)
plt.plot(ref_processed_data['minutes'], normalized_aligned_smooth, label='归一化对齐数据', linestyle='--')

plt.xlabel('时间 (分钟)')
plt.ylabel('人流量')
plt.title(f'{TEST_STATION}站 DTW 对齐效果')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()