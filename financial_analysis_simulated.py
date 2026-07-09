import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持（Windows/macOS/Linux 通用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ================== 1. 生成模拟股票价格（随机游走） ==================
print("="*50)
print("1. 生成模拟股票价格")
n_days = 252          # 一年交易日
n_stocks = 5
initial_prices = np.array([100, 150, 200, 80, 120])

# 生成每日对数收益率（均值0.0005，标准差0.02）
daily_log_returns = np.random.normal(loc=0.0005, scale=0.02, size=(n_days, n_stocks))
# 累积收益率并计算价格
cumulative_log = np.cumsum(daily_log_returns, axis=0)
prices = initial_prices * np.exp(cumulative_log)
# 将初始价格插入第一行
prices = np.vstack([initial_prices, prices])

print("价格矩阵形状:", prices.shape)
print("前5天价格（股票1）:", prices[:5, 0])

# ================== 2. 计算收益率与波动率 ==================
print("\n" + "="*50)
print("2. 收益率与波动率")
returns = np.diff(prices, axis=0) / prices[:-1]   # 日收益率

for i in range(n_stocks):
    mean_ret = np.mean(returns[:, i])
    std_ret = np.std(returns[:, i])
    print(f"股票{i+1}: 平均日收益率={mean_ret:.5f}, 日波动率={std_ret:.5f}")

# ================== 3. 移动平均线（使用 NumPy 实现） ==================
print("\n" + "="*50)
print("3. 移动平均线（股票1，20日与50日）")
def sma_numpy(prices, window):
    """返回与输入等长的数组，前window-1个为NaN"""
    sma = np.full_like(prices, np.nan, dtype=float)
    for i in range(window - 1, len(prices)):
        sma[i] = np.mean(prices[i - window + 1 : i + 1])
    return sma

stock1_prices = prices[:, 0]
sma_20 = sma_numpy(stock1_prices, 20)
sma_50 = sma_numpy(stock1_prices, 50)
print("20日均线（最后5个有效值）:", sma_20[-5:])
print("50日均线（最后5个有效值）:", sma_50[-5:])

# ================== 4. 投资组合风险分析 ==================
print("\n" + "="*50)
print("4. 投资组合分析（等权重）")
weights = np.ones(n_stocks) / n_stocks
cov_matrix = np.cov(returns.T)
print("协方差矩阵:\n", cov_matrix)
corr_matrix = np.corrcoef(returns.T)
print("相关系数矩阵:\n", corr_matrix)

# 投资组合年化收益与风险
port_returns = np.dot(returns, weights)
annual_return = np.mean(port_returns) * 252
annual_vol = np.std(port_returns) * np.sqrt(252)
print(f"等权重组合年化收益率: {annual_return:.4f}")
print(f"等权重组合年化波动率: {annual_vol:.4f}")

# 夏普比率（无风险利率2%）
sharpe = (annual_return - 0.02) / annual_vol
print(f"夏普比率: {sharpe:.4f}")

# ================== 5. 可视化 ==================
print("\n" + "="*50)
print("5. 生成可视化图表 (保存为 analysis.png)")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：股价走势
ax1 = axes[0, 0]
for i in range(n_stocks):
    ax1.plot(prices[:, i], label=f'股票{i+1}')
ax1.set_title('模拟股票价格（随机游走）')
ax1.set_xlabel('交易日')
ax1.set_ylabel('价格')
ax1.legend()
ax1.grid(alpha=0.3)

# 子图2：移动平均线（股票1）
ax2 = axes[0, 1]
ax2.plot(stock1_prices, label='股价', linewidth=1.5)
ax2.plot(sma_20, label='20日均线', linestyle='--')
ax2.plot(sma_50, label='50日均线', linestyle='--')
ax2.set_title('股票1 与移动平均线')
ax2.set_xlabel('交易日')
ax2.set_ylabel('价格')
ax2.legend()
ax2.grid(alpha=0.3)

# 子图3：收益率分布
ax3 = axes[1, 0]
for i in range(n_stocks):
    ax3.hist(returns[:, i], bins=50, alpha=0.4, label=f'股票{i+1}')
ax3.set_title('日收益率分布')
ax3.set_xlabel('日收益率')
ax3.set_ylabel('频数')
ax3.legend()
ax3.grid(alpha=0.3)

# 子图4：投资组合累计收益
ax4 = axes[1, 1]
cumulative_port = np.cumprod(1 + port_returns) - 1
ax4.plot(cumulative_port, color='blue', linewidth=2)
ax4.set_title('等权重组合累计收益率')
ax4.set_xlabel('交易日')
ax4.set_ylabel('累计收益率')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analysis.png', dpi=150)
plt.show()
print("图表已保存为 analysis.png")