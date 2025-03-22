import numpy as np
from dtaidistance import dtw
from scipy.stats import entropy


def generate_diverse_poi(poi, target_shape, num_candidates=20):
    """
    生成与原始 POI 数据差异较大的新 POI 张量
    :param poi: 原始 POI 张量 (N, M, K)
    :param target_shape: 目标形状 (N_new, M_new, K_new)
    :param num_candidates: 候选方案数量
    :return: 差异最大的候选 POI 张量
    """
    N, M, K = poi.shape
    N_new, M_new, K_new = target_shape
    best_diff = -np.inf
    best_poi = None

    # 分析原始 POI 特征
    poi_density = np.sum(poi) / (N * M * K)  # POI 密度
    is_single_center = check_single_center(poi)

    for _ in range(num_candidates):
        # 生成候选 POI 张量
        if is_single_center:
            candidate = generate_multi_center_poi(N_new, M_new, K_new)
        else:
            candidate = generate_hybrid_poi(N_new, M_new, K_new)

        # 调整 POI 密度以匹配原始数据
        candidate = candidate * (poi_density / np.mean(candidate))

        # 有效性检查
        if not is_valid_poi(candidate):
            continue

        # 计算差异
        current_diff = calculate_difference(poi, candidate)

        if current_diff > best_diff:
            best_diff = current_diff
            best_poi = candidate

    return best_poi if best_poi is not None else poi


def generate_multi_center_poi(N, M, K):
    """生成多中心 POI 分布"""
    poi = np.zeros((N, M, K))
    centers = [N // 4, N // 2, 3 * N // 4]  # 多中心设计

    for c in centers:
        for i in range(N):
            for j in range(M):
                for k in range(K):
                    # 使用高斯分布生成 POI 数量
                    distance = abs(i - c)
                    poi[i, j, k] = np.random.normal(loc=100 / (distance + 1), scale=10)

    # 确保 POI 数量非负
    poi = np.maximum(poi, 0)
    return poi


def generate_hybrid_poi(N, M, K):
    """生成混合型 POI 分布"""
    poi = np.zeros((N, M, K))

    # 基础分布（单中心）
    center = N // 2
    for i in range(N):
        for j in range(M):
            for k in range(K):
                distance = abs(i - center)
                poi[i, j, k] = np.random.normal(loc=50 / (distance + 1), scale=5)

    # 添加随机热点
    hotspots = np.random.randint(0, N, size=5)
    for h in hotspots:
        poi[h, :, :] *= 2  # 热点区域 POI 数量翻倍

    # 确保 POI 数量非负
    poi = np.maximum(poi, 0)
    return poi


def check_single_center(poi):
    """检查是否单中心分布"""
    N, M, K = poi.shape
    center = N // 2
    center_density = np.sum(poi[center, :, :])
    outer_density = np.sum(poi) - center_density
    return center_density > outer_density  # 中心密度高于外围


def is_valid_poi(poi):
    """验证 POI 张量有效性"""
    # 检查 POI 数量非负
    if np.any(poi < 0):
        return False
    return True


def calculate_difference(orig, candidate):
    """计算差异值（DTW + KL）"""
    # 将 POI 张量展平为序列
    orig_flat = orig.flatten()
    cand_flat = candidate.flatten()

    # DTW距离计算
    dtw_dist = dtw.distance(orig_flat, cand_flat)

    # KL散度计算（POI 数量分布）
    orig_p = orig_flat / np.sum(orig_flat)
    cand_p = cand_flat / np.sum(cand_flat)
    kl_div = entropy(orig_p, cand_p)

    return dtw_dist + kl_div


# 示例使用
if __name__ == "__main__":
    # 生成示例 POI 张量
    N, M, K = 10, 10, 10
    original_poi = generate_multi_center_poi(N, M, K)

    # 生成差异 POI 张量
    target_shape = (10, 10, 10)  # 目标形状
    diversified_poi = generate_diverse_poi(original_poi, target_shape)

    print("原始 POI 特征:")
    print(f"- POI 密度: {np.sum(original_poi) / (N * M * K):.2f}")
    print(f"- 中心区域 POI 数量: {np.sum(original_poi[N // 2, :, :])}")

    print("\n生成 POI 特征:")
    print(f"- POI 密度: {np.sum(diversified_poi) / np.prod(target_shape):.2f}")
    print(f"- 中心区域 POI 数量: {np.sum(diversified_poi[target_shape[0] // 2, :, :])}")
    print(f"- DTW+KL差异值: {calculate_difference(original_poi, diversified_poi):.2f}")