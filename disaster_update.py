import random


def update_disasters(environment):
    """
    更新灾情点的状态，包括：
    1. 现有灾情点的紧急程度下降（模拟时间推移导致的衰减）。
    2. 可能产生新的灾情点（模拟余震或新的建筑倒塌）。
    3. 模拟部分灾情点的延迟发现（信息不立即被报告）。

    :param environment: Environment 对象，包含城市地图、灾情点和救援人员信息
    """
    # 1. 遍历所有灾情点，降低紧急程度
    for point, data in list(environment.disasters.items()):  # 使用 list() 避免遍历时修改字典
        if data["level"] > 0:
            data["level"] -= random.randint(1, 2)  # 每个时间步减少 1~2 级
        if data["level"] <= 0:
            del environment.disasters[point]  # 如果灾情等级降为 0，移除该灾情点（任务完成或无存活者）

    # 2. 按一定概率生成新的灾情点
    if random.random() < 0.2:  # 20% 概率新增灾情点
        new_point = (random.randint(0, environment.GRID_SIZE - 1), random.randint(0, environment.GRID_SIZE - 1))
        if new_point not in environment.disasters:  # 确保新灾情点未被占用
            environment.disasters[new_point] = {
                "level": random.randint(5, 10),  # 新灾情点的紧急程度
                "rescue_needed": random.randint(1, 5)  # 需要的救援人员数量
            }

    # 3. 模拟信息延迟，部分灾情点被延迟发现
    if random.random() < 0.1:  # 10% 概率延迟发现一个灾情点
        hidden_point = (random.randint(0, environment.GRID_SIZE - 1), random.randint(0, environment.GRID_SIZE - 1))
        if hidden_point not in environment.disasters:  # 确保不会重复
            environment.disasters[hidden_point] = {
                "level": random.randint(3, 7),  # 较低的灾情等级
                "rescue_needed": random.randint(1, 3)  # 需要的救援人员数量
            }

