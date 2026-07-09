import numpy as np

# ================== 1. 创建不同维度的数组 ==================
print("="*50)
print("1. 创建数组")
arr_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]],
                   [[9, 10], [11, 12]]])
print("1D:", arr_1d.shape, arr_1d.ndim, "D")
print("2D:\n", arr_2d)
print("3D:\n", arr_3d)

# ================== 2. 索引与切片 ==================
print("\n" + "="*50)
print("2. 索引与切片")
print("arr_1d[2:6]   =", arr_1d[2:6])
print("arr_1d[::2]   =", arr_1d[::2])
print("arr_1d[::-1]  =", arr_1d[::-1])          # 反转
print("arr_2d[:, 1]  =", arr_2d[:, 1])          # 第二列
print("arr_2d[1:, 2:]=\n", arr_2d[1:, 2:])
print("arr_3d[0]     =\n", arr_3d[0])
print("arr_3d[:, 0, 1]=", arr_3d[:, 0, 1])

# ================== 3. 形状变换 ==================
print("\n" + "="*50)
print("3. 形状变换")
arr_flat = arr_2d.reshape(12)
print("展平:", arr_flat)
arr_reshaped = arr_flat.reshape(2, 6)
print("reshape(2,6):\n", arr_reshaped)
arr_transposed = arr_2d.T
print("转置:\n", arr_transposed)
# 添加新轴
arr_newaxis = arr_1d[np.newaxis, :]
print("添加轴后形状:", arr_newaxis.shape)

# ================== 4. 矩阵基本运算 ==================
print("\n" + "="*50)
print("4. 矩阵运算")
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8, 9],
              [10, 11, 12]])
print("A + B =\n", A + B)
print("A * B (元素级) =\n", A * B)
D = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print("A @ D =\n", A @ D)          # 矩阵乘法
print("A * 3 =\n", A * 3)

# ================== 5. 随机数据生成与统计分析 ==================
print("\n" + "="*50)
print("5. 随机数据与统计")
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=1000)
print("正态分布(1000个):")
print(f"  均值 = {np.mean(data):.4f}")
print(f"  标准差 = {np.std(data):.4f}")
print(f"  方差 = {np.var(data):.4f}")
print(f"  最小值/最大值 = {np.min(data):.4f} / {np.max(data):.4f}")
print("  25%, 50%, 75% 分位数:", np.percentile(data, [25, 50, 75]))

rand_mat = np.random.randint(1, 100, size=(4, 5))
print("\n随机整数矩阵:\n", rand_mat)
print("每行均值:", np.mean(rand_mat, axis=1))
print("每列均值:", np.mean(rand_mat, axis=0))
print("协方差矩阵:\n", np.cov(rand_mat, rowvar=False))