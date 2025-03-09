import heapq
import math

# 设定灾情分级临界值（CDT）
CRITICAL_DISASTER_THRESHOLD = 8  # 灾情等级大于等于 8 视为重大灾情

# 设定资源饱和度阈值（RSI）
RESOURCE_SATURATION_THRESHOLD = 1.0  # RSI ≥ 1.0 时不再增加救援人员


def calculate_rsi(disaster, assigned_rescuers):
    """
    计算灾情点的资源饱和度（RSI = 已分配资源 / 需求资源）。
    """
    if disaster["rescue_needed"] == 0:
        return 1.0  # 避免除零错误
    return assigned_rescuers / disaster["rescue_needed"]


def classify_disaster(disaster):
    """
    判断灾情级别（Minor / Moderate / Critical）。
    """
    if disaster["level"] < CRITICAL_DISASTER_THRESHOLD:
        return "Moderate" if disaster["level"] >= CRITICAL_DISASTER_THRESHOLD / 2 else "Minor"
    return "Critical"


def a_star_search(grid_size, start, goal):
    """
    使用 A* 算法计算最短路径，优化救援人员移动路径
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # 曼哈顿距离

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path  # 返回最短路径

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # 如果找不到路径，返回空路径


def hybrid_rescue_dispatch(rescuers, disasters, grid_size):
    """
    采用智能调度策略：
    1. 依据灾情等级，分配不同的调度策略：
       - 轻度灾情（Minor）：采用最近任务优先（NTF）。
       - 中等灾情（Moderate）：采用混合评分调度（HSD）。
       - 重大灾情（Critical）：采用最大任务优先（HDF）。
    2. 计算资源饱和度（RSI），避免资源浪费。
    3. 采用任务池（Task Pool）动态分配任务，避免重复指派。
    """

    # 过滤掉已冻结的灾情点，创建任务池
    task_pool = {}
    for point, data in disasters.items():
        # 跳过已经完成救援或自然结束的灾情点
        if data.get("frozen_rescue", False) or data.get("frozen_level", False):
            continue
        task_pool[point] = {"data": data, "assigned": 0}
    
    # 如果没有有效的灾情点，直接返回
    if not task_pool:
        return

    # 1️⃣ 先对重大灾情（Critical）执行最大任务优先（HDF）
    critical_disasters = sorted(
        [point for point, task in task_pool.items() if classify_disaster(task["data"]) == "Critical"],
        key=lambda p: disasters[p]["level"], reverse=True
    )

    # 2️⃣ 其次，对中等灾情（Moderate）执行混合评分调度（HSD）
    moderate_disasters = sorted(
        [point for point, task in task_pool.items() if classify_disaster(task["data"]) == "Moderate"],
        key=lambda p: disasters[p]["level"] / (min_distance_to_rescuer(p, rescuers, grid_size) + 1),
        reverse=True
    )

    # 3️⃣ 最后，对轻度灾情（Minor）执行最近任务优先（NTF）
    minor_disasters = sorted(
        [point for point, task in task_pool.items() if classify_disaster(task["data"]) == "Minor"],
        key=lambda p: min_distance_to_rescuer(p, rescuers, grid_size)
    )

    # 4️⃣ 依次为每个救援人员分配最优目标
    for rescuer in rescuers:
        best_target = None
        best_score = -1

        for priority_list in [critical_disasters, moderate_disasters, minor_disasters]:
            for target in priority_list:
                if target in task_pool:
                    task = task_pool[target]
                    rsi = calculate_rsi(task["data"], task["assigned"])

                    if rsi >= RESOURCE_SATURATION_THRESHOLD:
                        continue

                    # ✅ 使用 A* 计算最短路径，优化路径选择
                    path = a_star_search(grid_size, rescuer["position"], target)
                    distance = len(path) if path else min_distance_to_rescuer(target, rescuers, grid_size)
                    score = task["data"]["level"] / (distance + 1)

                    if score > best_score:
                        best_score = score
                        best_target = target

        # 分配任务
        if best_target:
            rescuer["target"] = best_target
            task_pool[best_target]["assigned"] += 1


def min_distance_to_rescuer(disaster_point, rescuers, grid_size):
    """
    计算灾情点到最近救援人员的最小距离。
    """
    return min(len(a_star_search(grid_size, rescuer["position"], disaster_point)) or
               abs(rescuer["position"][0] - disaster_point[0]) + abs(rescuer["position"][1] - disaster_point[1])
               for rescuer in rescuers)
