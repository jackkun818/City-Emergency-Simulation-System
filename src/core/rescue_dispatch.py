import heapq
import math
from . import config

# 使用配置文件中的常量
CRITICAL_DISASTER_THRESHOLD = config.CRITICAL_DISASTER_THRESHOLD  # 灾情等级大于等于此值视为重大灾情
RESOURCE_SATURATION_THRESHOLD = config.RESOURCE_SATURATION_THRESHOLD  # RSI ≥ 此值时不再增加救援人员


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
    # 考虑灾情等级与到达时间的比例，而不仅仅是距离
    moderate_disasters = []
    for point in [p for p, task in task_pool.items() if classify_disaster(task["data"]) == "Moderate"]:
        # 找到最快能到达该点的救援者
        min_arrival_time = float('inf')
        for rescuer in rescuers:
            if "target" in rescuer and rescuer["target"] is not None:
                continue  # 跳过已有任务的救援者
            
            path = a_star_search(grid_size, rescuer["position"], point)
            distance = len(path) if path else (abs(rescuer["position"][0] - point[0]) + abs(rescuer["position"][1] - point[1]))
            rescuer_speed = rescuer.get("speed", 1)
            arrival_time = distance / rescuer_speed if rescuer_speed > 0 else float('inf')
            min_arrival_time = min(min_arrival_time, arrival_time)
        
        # 计算评分：灾情等级 / 到达时间
        score = disasters[point]["level"] / (min_arrival_time + 1)
        moderate_disasters.append((point, score))
    
    # 按评分从高到低排序
    moderate_disasters = [p for p, _ in sorted(moderate_disasters, key=lambda x: x[1], reverse=True)]

    # 3️⃣ 最后，对轻度灾情（Minor）执行最近任务优先（NTF）
    # 考虑到达时间而不仅仅是距离
    minor_disasters = []
    for point in [p for p, task in task_pool.items() if classify_disaster(task["data"]) == "Minor"]:
        # 找到最快能到达该点的救援者
        min_arrival_time = float('inf')
        for rescuer in rescuers:
            if "target" in rescuer and rescuer["target"] is not None:
                continue  # 跳过已有任务的救援者
            
            path = a_star_search(grid_size, rescuer["position"], point)
            distance = len(path) if path else (abs(rescuer["position"][0] - point[0]) + abs(rescuer["position"][1] - point[1]))
            rescuer_speed = rescuer.get("speed", 1)
            arrival_time = distance / rescuer_speed if rescuer_speed > 0 else float('inf')
            min_arrival_time = min(min_arrival_time, arrival_time)
        
        minor_disasters.append((point, min_arrival_time))
    
    # 按到达时间从低到高排序
    minor_disasters = [p for p, _ in sorted(minor_disasters, key=lambda x: x[1])]

    # 4️⃣ 首先按照救援能力（capacity）对救援人员进行排序，能力强的优先分配
    sorted_rescuers = sorted(
        rescuers, 
        key=lambda r: r.get("capacity", 0), 
        reverse=True  # 能力强的排在前面
    )
    
    # 记录已分配的目标，避免多人前往同一灾情点
    assigned_targets = set()
    
    # 为每个救援人员分配最优目标，能力强的先选择
    for rescuer in sorted_rescuers:
        # 如果救援人员正在执行救援任务，跳过重新分配
        if rescuer.get("actively_rescuing", False):
            # 确保正在救援的目标被标记为已分配
            if "target" in rescuer and rescuer["target"] is not None:
                assigned_targets.add(rescuer["target"])
            continue
            
        # 如果已有目标且目标有效，则跳过
        if "target" in rescuer and rescuer["target"] is not None and rescuer["target"] in task_pool:
            assigned_targets.add(rescuer["target"])  # 标记此目标已被分配
            continue
            
        best_target = None
        best_score = -1

        # 按优先级依次考虑不同类型灾情
        for priority_list in [critical_disasters, moderate_disasters, minor_disasters]:
            for target in priority_list:
                # 跳过已分配的目标
                if target in assigned_targets:
                    continue
                    
                if target in task_pool:
                    task = task_pool[target]
                    rsi = calculate_rsi(task["data"], task["assigned"])

                    if rsi >= RESOURCE_SATURATION_THRESHOLD:
                        continue

                    # ✅ 使用 A* 计算最短路径，优化路径选择
                    path = a_star_search(grid_size, rescuer["position"], target)
                    distance = len(path) if path else min_distance_to_rescuer(target, rescuers, grid_size)
                    
                    # 获取救援者的移动速度，默认为1
                    rescuer_speed = rescuer.get("speed", 1)
                    
                    # 计算到达时间 = 距离 / 速度
                    arrival_time = distance / rescuer_speed if rescuer_speed > 0 else float('inf')
                    
                    # 综合考虑灾情等级、能力匹配度和到达时间
                    # 计算能力匹配度：灾情点的rescue_needed与救援员capacity的比例，越接近1越好
                    capacity_match = min(
                        rescuer.get("capacity", 1) / task["data"].get("rescue_needed", 1),
                        1.0
                    )
                    
                    # 新的评分公式：结合灾情等级、能力匹配度和到达时间
                    # 到达时间越短，评分越高
                    score = (task["data"]["level"] * capacity_match) / (arrival_time*2 + 1)

                    if score > best_score:
                        best_score = score
                        best_target = target

        # 分配任务
        if best_target:
            rescuer["target"] = best_target
            task_pool[best_target]["assigned"] += 1
            assigned_targets.add(best_target)  # 标记此目标已被分配


def min_distance_to_rescuer(disaster_point, rescuers, grid_size):
    """
    计算灾情点到最近救援人员的最小距离。
    """
    return min(len(a_star_search(grid_size, rescuer["position"], disaster_point)) or
               abs(rescuer["position"][0] - disaster_point[0]) + abs(rescuer["position"][1] - disaster_point[1])
               for rescuer in rescuers)
