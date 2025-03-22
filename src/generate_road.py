from dtaidistance import dtw  # 正确导入方式
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import entropy


def generate_diverse_road(r, num_candidates=20):
    """
    生成与原始道路网络差异最大的新路网
    :param r: 原始邻接矩阵 (N x N)
    :param num_candidates: 候选方案数量
    :return: 差异最大的候选邻接矩阵
    """
    N = r.shape[0]
    best_diff = -np.inf
    best_r = None

    # 分析原始网络特征
    edge_density = np.sum(r) / (N * (N - 1))  # 排除对角线
    degree_dist = np.sum(r, axis=0)
    is_grid_like = check_grid_like(r)

    for _ in range(num_candidates):
        # 生成候选网络
        if is_grid_like and edge_density > 0.3:
            candidate = generate_radial_network(N)
        else:
            candidate = generate_hybrid_network(N)

        # 有效性检查
        if not is_valid_road(candidate):
            continue

        # 计算差异
        current_diff = calculate_difference(r, candidate)

        if current_diff > best_diff:
            best_diff = current_diff
            best_r = candidate

    return best_r if best_r is not None else r


def generate_radial_network(N):
    """生成放射状路网"""
    r = np.zeros((N, N))
    centers = [0, N // 4, N // 2]  # 多中心设计

    # 创建主干道
    for c in centers:
        for i in range(N):
            if i != c and np.random.rand() < 0.7:
                r[c, i] = r[i, c] = 1

    # 添加环状连接
    for i in range(N - 1):
        if i not in centers and i + 1 not in centers:
            r[i, i + 1] = r[i + 1, i] = 1
    return r


def generate_hybrid_network(N):
    """生成混合型路网"""
    r = np.zeros((N, N))

    # 创建基础网格
    grid_size = int(np.sqrt(N))
    if grid_size ** 2 == N:
        for i in range(N):
            if i % grid_size != grid_size - 1:
                r[i, i + 1] = r[i + 1, i] = 1
            if i < N - grid_size:
                r[i, i + grid_size] = r[i + grid_size, i] = 1

    # 随机添加对角线连接
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() < 0.05:
                r[i, j] = r[j, i] = 1
    return r


def check_grid_like(matrix):
    """检查是否类似网格结构"""
    N = matrix.shape[0]
    grid_size = int(np.sqrt(N))
    if grid_size ** 2 != N:
        return False

    correct_edges = 0
    for i in range(N):
        right = i + 1 if (i + 1) % grid_size != 0 else -1
        down = i + grid_size if i + grid_size < N else -1

        if right != -1 and matrix[i, right] == 1:
            correct_edges += 1
        if down != -1 and matrix[i, down] == 1:
            correct_edges += 1

    return correct_edges / (2 * N - 2 * grid_size) > 0.8  # 80%符合网格特征


def is_valid_road(matrix):
    """验证路网有效性"""
    # 检查连通性
    num_components, _ = connected_components(matrix, directed=False)
    if num_components > 1:  # 连通分量数量大于1表示不连通
        return False

    # 检查节点度数（至少2条连接）
    degrees = np.sum(matrix, axis=0)
    if np.any(degrees < 2):
        return False

    return True


def calculate_difference(orig, candidate):
    """计算差异值（DTW + KL）"""
    # DTW距离计算
    dtw_dist = dtw.distance(orig.flatten(), candidate.flatten())  # 使用 dtw.distance

    # KL散度计算（度数分布）
    orig_deg = np.sum(orig, axis=0) + 1e-10
    cand_deg = np.sum(candidate, axis=0) + 1e-10

    orig_p = orig_deg / np.sum(orig_deg)
    cand_p = cand_deg / np.sum(cand_deg)

    kl_div = entropy(orig_p, cand_p)

    return dtw_dist + kl_div


# 示例使用
if __name__ == "__main__":
    # 生成示例网格网络
    N = 25
    original = generate_hybrid_network(N)

    # 生成差异网络
    diversified = generate_diverse_road(original)

    print("原始网络特征:")
    print(f"- 边密度: {np.sum(original) / N / (N - 1):.2f}")
    print(f"- 平均度数: {np.mean(np.sum(original, axis=0)):.2f}")

    print("\n生成网络特征:")
    print(f"- 边密度: {np.sum(diversified) / N / (N - 1):.2f}")
    print(f"- 平均度数: {np.mean(np.sum(diversified, axis=0)):.2f}")
    print(f"- DTW+KL差异值: {calculate_difference(original, diversified):.2f}")