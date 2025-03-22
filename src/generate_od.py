import numpy as np
from dtaidistance import dtw
from scipy.stats import entropy


def generate_diverse_od(M, N, num_candidates=20):
    """
    生成与原始 OD 数据差异较大的新 OD 矩阵
    :param M: 城市行数
    :param N: 城市列数
    :param num_candidates: 候选方案数量
    :return: 差异最大的候选 OD 矩阵
    """
    MN = M * N
    best_diff = -np.inf
    best_od = None

    # 分析原始 OD 特征（假设原始 OD 矩阵为全零）
    original_od = np.zeros((MN, MN))
    is_single_center = check_single_center(original_od)

    for _ in range(num_candidates):
        # 生成候选 OD 矩阵
        if is_single_center:
            candidate = generate_multi_center_od(M, N)
        else:
            candidate = generate_hybrid_od(M, N)

        # 有效性检查
        if not is_valid_od(candidate):
            continue

        # 计算差异
        current_diff = calculate_difference(original_od, candidate)

        if current_diff > best_diff:
            best_diff = current_diff
            best_od = candidate

    return best_od if best_od is not None else original_od


def generate_multi_center_od(M, N):
    """生成多中心 OD 矩阵"""
    MN = M * N
    od = np.zeros((MN, MN))
    centers = [MN // 4, MN // 2, 3 * MN // 4]  # 多中心设计

    for c in centers:
        for i in range(MN):
            for j in range(MN):
                # 使用距离衰减模型生成流量
                distance = abs(i - c) + abs(j - c)  # 曼哈顿距离
                od[i, j] = np.random.normal(loc=100 / (distance + 1), scale=10)

    # 确保流量非负
    od = np.maximum(od, 0)
    return od


def generate_hybrid_od(M, N):
    """生成混合型 OD 矩阵"""
    MN = M * N
    od = np.zeros((MN, MN))

    # 基础分布（单中心）
    center = MN // 2
    for i in range(MN):
        for j in range(MN):
            distance = abs(i - center) + abs(j - center)  # 曼哈顿距离
            od[i, j] = np.random.normal(loc=50 / (distance + 1), scale=5)

    # 添加随机热点
    hotspots = np.random.randint(0, MN, size=5)
    for h in hotspots:
        od[h, :] *= 2  # 热点区域流量翻倍

    # 确保流量非负
    od = np.maximum(od, 0)
    return od


def check_single_center(od):
    """检查是否单中心分布"""
    MN = od.shape[0]
    center = MN // 2
    center_density = np.sum(od[center, :])
    outer_density = np.sum(od) - center_density
    return center_density > outer_density  # 中心密度高于外围


def is_valid_od(od):
    """验证 OD 矩阵有效性"""
    # 检查流量非负
    if np.any(od < 0):
        return False
    return True


def calculate_difference(orig, candidate):
    """计算差异值（DTW + KL）"""
    # 将 OD 矩阵展平为序列
    orig_flat = orig.flatten()
    cand_flat = candidate.flatten()

    # DTW距离计算
    dtw_dist = dtw.distance(orig_flat, cand_flat)

    # KL散度计算（流量分布）
    orig_sum = np.sum(orig_flat)
    cand_sum = np.sum(cand_flat)

    # 避免除以零
    if orig_sum == 0:
        orig_p = np.zeros_like(orig_flat)
    else:
        orig_p = orig_flat / orig_sum

    if cand_sum == 0:
        cand_p = np.zeros_like(cand_flat)
    else:
        cand_p = cand_flat / cand_sum

    # 确保 orig_p 和 cand_p 的形状一致
    min_length = min(len(orig_p), len(cand_p))
    orig_p = orig_p[:min_length]
    cand_p = cand_p[:min_length]

    kl_div = entropy(orig_p, cand_p)

    return dtw_dist + kl_div


# 示例使用
if __name__ == "__main__":
    # 生成示例 OD 矩阵
    M, N = 5, 5  # 城市大小为 5x5
    original_od = np.zeros((M * N, M * N))  # 假设原始 OD 矩阵为全零

    # 生成差异 OD 矩阵
    diversified_od = generate_diverse_od(M, N)

    print("原始 OD 特征:")
    print(f"- OD 矩阵大小: {original_od.shape}")
    print(f"- 总流量: {np.sum(original_od):.2f}")

    print("\n生成 OD 特征:")
    print(f"- OD 矩阵大小: {diversified_od.shape}")
    print(f"- 总流量: {np.sum(diversified_od):.2f}")
    print(f"- DTW+KL差异值: {calculate_difference(original_od, diversified_od):.2f}")