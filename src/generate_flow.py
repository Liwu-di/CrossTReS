import numpy as np
from dtaidistance import dtw
from scipy.stats import entropy


def generate_diverse_flow(flow, target_shape, num_candidates=20):
    """
    生成与原始流量数据差异较大的新流量张量
    :param flow: 原始流量张量 (N, T, M)
    :param target_shape: 目标形状 (N_new, T_new, M_new)
    :param num_candidates: 候选方案数量
    :return: 差异最大的候选流量张量
    """
    N, T, M = flow.shape
    N_new, T_new, M_new = target_shape
    best_diff = -np.inf
    best_flow = None

    # 分析原始流量特征
    flow_density = np.sum(flow) / (N * T * M)  # 流量密度
    is_single_center = check_single_center(flow)

    for _ in range(num_candidates):
        # 生成候选流量张量
        if is_single_center:
            candidate = generate_multi_center_flow(N_new, T_new, M_new)
        else:
            candidate = generate_hybrid_flow(N_new, T_new, M_new)

        # 调整流量密度以匹配原始数据
        candidate = candidate * (flow_density / np.mean(candidate))

        # 有效性检查
        if not is_valid_flow(candidate):
            continue

        # 计算差异
        current_diff = calculate_difference(flow, candidate)

        if current_diff > best_diff:
            best_diff = current_diff
            best_flow = candidate

    return best_flow if best_flow is not None else flow


def generate_multi_center_flow(N, T, M):
    """生成多中心流量分布"""
    flow = np.zeros((N, T, M))
    centers = [N // 4, N // 2, 3 * N // 4]  # 多中心设计

    for c in centers:
        for i in range(N):
            for j in range(T):
                for k in range(M):
                    # 使用高斯分布生成流量
                    distance = abs(i - c)
                    flow[i, j, k] = np.random.normal(loc=100 / (distance + 1), scale=10)

    # 确保流量非负
    flow = np.maximum(flow, 0)
    return flow


def generate_hybrid_flow(N, T, M):
    """生成混合型流量分布"""
    flow = np.zeros((N, T, M))

    # 基础分布（单中心）
    center = N // 2
    for i in range(N):
        for j in range(T):
            for k in range(M):
                distance = abs(i - center)
                flow[i, j, k] = np.random.normal(loc=50 / (distance + 1), scale=5)

    # 添加随机热点
    hotspots = np.random.randint(0, N, size=5)
    for h in hotspots:
        flow[h, :, :] *= 2  # 热点区域流量翻倍

    # 确保流量非负
    flow = np.maximum(flow, 0)
    return flow


def check_single_center(flow):
    """检查是否单中心分布"""
    N, T, M = flow.shape
    center = N // 2
    center_density = np.sum(flow[center, :, :])
    outer_density = np.sum(flow) - center_density
    return center_density > outer_density  # 中心密度高于外围


def is_valid_flow(flow):
    """验证流量张量有效性"""
    # 检查流量非负
    if np.any(flow < 0):
        return False
    return True


def calculate_difference(orig, candidate):
    """计算差异值（DTW + KL）"""
    # 将流量张量展平为序列
    orig_flat = orig.flatten()
    cand_flat = candidate.flatten()

    # DTW距离计算
    dtw_dist = dtw.distance(orig_flat, cand_flat)

    # KL散度计算（流量分布）
    orig_p = orig_flat / np.sum(orig_flat)
    cand_p = cand_flat / np.sum(cand_flat)

    # 确保 orig_p 和 cand_p 的形状一致
    min_length = min(len(orig_p), len(cand_p))
    orig_p = orig_p[:min_length]
    cand_p = cand_p[:min_length]

    kl_div = entropy(orig_p, cand_p)

    return dtw_dist + kl_div


# 示例使用
if __name__ == "__main__":
    # 生成示例流量张量
    N, T, M = 10, 24, 3  # 10个区域，24小时，3种交通模式
    original_flow = generate_multi_center_flow(N, T, M)

    # 生成差异流量张量
    target_shape = (15, 24, 4)  # 目标形状
    diversified_flow = generate_diverse_flow(original_flow, target_shape)

    print("原始流量特征:")
    print(f"- 流量密度: {np.sum(original_flow) / (N * T * M):.2f}")
    print(f"- 中心区域流量: {np.sum(original_flow[N // 2, :, :])}")

    print("\n生成流量特征:")
    print(f"- 流量密度: {np.sum(diversified_flow) / np.prod(target_shape):.2f}")
    print(f"- 中心区域流量: {np.sum(diversified_flow[target_shape[0] // 2, :, :])}")
    print(f"- DTW+KL差异值: {calculate_difference(original_flow, diversified_flow):.2f}")