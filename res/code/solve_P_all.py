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
REFERENCE_STATION = "五和"  # 参考站点名称
DATA_DIR = "output_stations" # 数据文件目录
LOESS_FRAC = 0.2          # LOESS 平滑参数，控制平滑度

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

def base_pattern(t, A, mu, sigma, alpha, beta):
    """定义的数学模型函数"""
    gaussian = A * np.exp(-(t - mu)**2 / (2 * sigma**2))
    asymmetry = 1 + alpha * np.tanh(beta * (t - mu))
    return gaussian * asymmetry

# 加载参考站数据
ref_raw_data, ref_processed_data = load_and_process(REFERENCE_STATION)
if ref_raw_data is None:
    exit()

all_station_names = [filename[:-4] for filename in os.listdir(DATA_DIR) if filename.endswith(".txt")]
fitted_curves = []
all_optimal_params = []

for test_station_name in all_station_names:
    if test_station_name == REFERENCE_STATION: # Skip reference station itself for DTW and fitting in loop
        continue

    test_raw_data, test_processed_data = load_and_process(test_station_name)
    if test_raw_data is None:
        continue

    # --- DTW 对齐 ---
    distance, path = fastdtw(ref_processed_data['smooth_normalized'].values.reshape(-1, 1),
                           test_processed_data['smooth_normalized'].values.reshape(-1, 1),
                           dist=euclidean)

    aligned_test_smooth = np.zeros_like(ref_processed_data['smooth'].values)
    alignment_counts = np.zeros_like(ref_processed_data['smooth'].values, dtype=float)

    for i, j in path:
        if i < len(aligned_test_smooth) and j < len(test_processed_data['smooth'].values):
            aligned_test_smooth[i] += test_processed_data['smooth'].values[j]
            alignment_counts[i] += 1

    aligned_test_smooth = np.divide(aligned_test_smooth, alignment_counts, out=np.zeros_like(aligned_test_smooth), where=alignment_counts != 0)

    # --- 数学模型拟合 ---
    time_points = ref_processed_data['minutes'].values
    average_pattern = aligned_test_smooth

    initial_guess = [1, 720, 180, 0.2, 0.05]
    try:
        optimal_params, _ = curve_fit(base_pattern,
                                     time_points,
                                     average_pattern,
                                     p0=initial_guess,
                                     method='lm',
                                     maxfev=10000)
        all_optimal_params.append(optimal_params) # Store parameters for each station
        fitted_curve = base_pattern(time_points, *optimal_params)
        fitted_curves.append(fitted_curve)
    except RuntimeError as e:
        print(f"警告: {test_station_name}站模型拟合失败: {e}, 使用初始猜测值")
        all_optimal_params.append(initial_guess) # Store initial guess if fitting fails
        fitted_curve = base_pattern(time_points, *initial_guess)
        fitted_curves.append(fitted_curve)


# --- 计算平均拟合曲线 ---
normalized_fitted_curves = [min_max_normalize(curve) for curve in fitted_curves]
average_fitted_curve = np.mean(normalized_fitted_curves, axis=0)

# --- 拟合平均拟合曲线 ---
time_points = ref_processed_data['minutes'].values
average_pattern_for_fit = average_fitted_curve
initial_guess_average = [1, 720, 180, 0.2, 0.05]

try:
    average_optimal_params, _ = curve_fit(base_pattern,
                                         time_points,
                                         average_pattern_for_fit,
                                         p0=initial_guess_average,
                                         method='lm',
                                         maxfev=10000)
except RuntimeError as e:
    print(f"警告: 平均拟合曲线模型拟合失败: {e}, 使用初始猜测值")
    average_optimal_params = initial_guess_average

# --- 可视化平均拟合曲线和拟合模型 ---
plt.figure(figsize=(12, 6))
plt.plot(time_points, average_fitted_curve, label='平均归一化拟合曲线', alpha=0.7, linewidth=2)
plt.plot(time_points, base_pattern(time_points, *average_optimal_params), label='平均拟合曲线模型', linestyle='--', color='red', linewidth=2)

plt.xlabel('时间 (分钟)')
plt.ylabel('归一化人流量')
plt.title('所有站点平均入站模式拟合')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --- 输出平均拟合参数 ---
param_names = ['A (幅值)', 'μ (峰值时间)', 'σ (宽度)', 'α (不对称性)', 'β (陡峭度)']
print("\n平均模型拟合参数：")
for name, value in zip(param_names, average_optimal_params):
    print(f"{name}: {value:.4f}")
