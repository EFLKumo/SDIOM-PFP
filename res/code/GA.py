import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

A = float(input("A = "))
mu = float(input("mu = "))
sigma = float(input("sigma = "))
alpha = float(input("alpha = "))
beta = float(input("beta = "))

# 定义乘客到达率函数 P(t)
def P(t):
    exponent = -((t - mu)**2) / (2 * (sigma**2))
    gaussian_part = A * math.exp(exponent)
    tanh_part = 1 + alpha * math.tanh(beta * (t - mu))
    P_t = gaussian_part * tanh_part
    return P_t

# 计算成本函数（动态归一化）
def compute_cost(T, t_current, gamma, eta):
    if T <= 0:
        return float('inf')
    d_tau = 0.01  # 积分步长（分钟）
    tau_values = np.arange(0, T, d_tau)
    actual_times = t_current + tau_values
    p_values = np.vectorize(P)(actual_times)
    
    integrand = p_values * (T - tau_values)
    integral = np.sum(integrand) * d_tau
    total_cost = (gamma * integral + eta) / T
    return total_cost

# 遗传算法优化
def genetic_algorithm(t_current, gamma, eta, population_size=100, generations=100):
    T_min = 2.0  # 最小发车间隔（分钟）
    T_max = 15.0  # 最大发车间隔（分钟）
    mutation_rate = 0.3
    mutation_std = 0.3

    # 初始化种群
    population = np.random.uniform(low=T_min, high=T_max, size=population_size)

    best_fitness_history = []
    best_T_history = []

    for generation in range(generations):
        # 计算适应度（1/Cost）
        costs = np.array([compute_cost(T, t_current, gamma, eta) for T in population])
        fitness = 1 / (costs + 1e-8)  # 避免除以零
        
        # 记录最佳个体
        best_idx = np.argmax(fitness)
        best_T = population[best_idx]
        best_cost = costs[best_idx]
        best_T_history.append(best_T)
        best_fitness_history.append(1 / best_cost)
        print(f"Generation {generation}: Best T = {best_T:.2f} mins, Cost = {best_cost:.2f}")
        
        # 轮盘赌选择
        probabilities = fitness / np.sum(fitness)
        parent_indices = np.random.choice(population_size, size=population_size, p=probabilities)
        parents = population[parent_indices]
        
        # 算术交叉（加权平均）
        offspring = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[(i+1)%population_size]
            alpha = np.random.rand()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = (1 - alpha) * p1 + alpha * p2
            offspring.extend([child1, child2])
        offspring = np.array(offspring[:population_size])
        
        # 高斯变异
        mask = np.random.rand(population_size) < mutation_rate
        noise = np.random.normal(0, mutation_std, population_size)
        offspring = np.where(mask, offspring + noise, offspring)
        offspring = np.clip(offspring, T_min, T_max)  # 限制在范围内
        
        population = offspring

    # 返回最优解
    best_T = best_T_history[-1]
    return best_T, best_T_history, best_fitness_history

# 可视化函数
def plot_system_status(t_current, best_T):
    plt.figure(figsize=(15, 10))
    
    # 子图1
    plt.subplot(1,2,1)
    t_range = np.linspace(0, 12*60, 1000)
    P_values = np.vectorize(P)(t_range)
    plt.plot(t_range, P_values, 'b-', label='P(t)')
    plt.axvline(x=t_current, color='r', linestyle='--', label='当前时间')
    plt.axvline(x=(t_current + best_T), color='g', linestyle='--', label='发车时间')
    plt.xlabel('分钟')
    plt.ylabel('人数')
    plt.title(f'乘客到达图像')
    plt.legend()
    plt.grid(True)

    # 子图2：成本函数曲线
    plt.subplot(1,2,2)
    T_range = np.linspace(2, 15, 100)
    costs = [compute_cost(T, t_current, gamma, eta) for T in T_range]
    plt.plot(T_range, costs, 'g-', label='Cost Curve')
    plt.axvline(x=best_T, color='r', label=f'Optimal T = {best_T:.1f}min')
    plt.xlabel('发车间隔（分钟）')
    plt.ylabel('总成本')
    plt.title('成本函数')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 参数设置
    current_time = float(input("模拟时间："))
    gamma = 2.0         # 等待时间成本系数
    eta = 10.0          # 发车固定成本

    # 运行遗传算法
    best_T, best_T_history, best_fitness_history = genetic_algorithm(current_time, gamma, eta)

    # 输出结果
    print(f"\n{best_T:.2f} 分钟后发车")

    # 可视化
    plot_system_status(current_time, best_T)