import heapq
import time

def a_star_search(grid_size, start, goal):
    """
    使用 A* 算法计算最短路径，避免救援人员绕路，提高救援效率。

    :param grid_size: 城市网格大小 (GRID_SIZE, GRID_SIZE)
    :param start: 救援人员起始坐标 (x, y)
    :param goal: 目标灾情点坐标 (x, y)
    :return: 经过 A* 计算后的路径（列表格式）
    """

    def heuristic(a, b):
        """ 使用曼哈顿距离作为启发式函数 """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))  # (优先级, 坐标)

    came_from = {}  # 记录路径
    g_score = {start: 0}  # g(n): 从起点到当前节点的路径成本
    f_score = {start: heuristic(start, goal)}  # f(n) = g(n) + h(n)

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
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上、下、左、右四个方向
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:  # 确保不超出地图边界
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # 如果找不到路径，返回空路径


def execute_rescue(rescuers, disasters, grid_size):
    """
    让救援人员按照 A* 规划路径前往目标灾情点并执行救援任务。

    :param rescuers: 救援人员列表，每个救援人员包含 {"id", "position", "speed", "capacity", "target"}
    :param disasters: 当前所有灾情点，格式：{(x, y): {"level": 10, "rescue_needed": 5}}
    :param grid_size: 城市地图网格大小
    """
    # 创建已完成救援的灾情点列表，避免在遍历过程中直接删除字典元素
    completed_disasters = []
    
    for rescuer in rescuers:
        # 确保救援人员有目标
        if "target" in rescuer and rescuer["target"] is not None and rescuer["target"] in disasters:
            target_x, target_y = rescuer["target"]
            x, y = rescuer["position"]

            # ✅ 计算最优路径
            path = a_star_search(grid_size, (x, y), (target_x, target_y))
            # 保存路径用于可视化
            rescuer["path"] = path

            # ✅ 处理救援人员速度
            if path:
                move_steps = min(rescuer.get("speed", 1), len(path))  # 限制移动步数
                rescuer["position"] = path[move_steps - 1]  # 走 `speed` 步
                # 记录救援人员的活动时间
                rescuer["active_time"] = rescuer.get("active_time", 0) + 1

            # ✅ 当救援人员抵达灾情点时，执行救援
            if rescuer["position"] == (target_x, target_y):
                # 先检查 `capacity` 是否存在
                if "capacity" not in rescuer:
                    print(f"❌ 错误: 救援人员 {rescuer['id']} 缺少 `capacity`，请检查 `environment.py`")
                    continue

                # 进行救援 - 只减少rescue_needed，不减少level
                disasters[(target_x, target_y)]["rescue_needed"] -= 1

                # ✅ 确保rescue_needed不会小于0
                disasters[(target_x, target_y)]["rescue_needed"] = max(0, disasters[(target_x, target_y)]["rescue_needed"])

                print(f"🚑 救援人员 {rescuer['id']} 在 {target_x, target_y} 进行救援，"
                      f"剩余等级: {disasters[(target_x, target_y)]['level']}，"
                      f"剩余需要救援: {disasters[(target_x, target_y)]['rescue_needed']}")

                # ✅ 判定救援是否完成 - 如果rescue_needed=0，表示成功救援
                if disasters[(target_x, target_y)]["rescue_needed"] <= 0:
                    print(f"✅ 灾情点 {target_x, target_y} 成功救援完成！")
                    disasters[(target_x, target_y)]["frozen_rescue"] = True
                    disasters[(target_x, target_y)]["rescue_success"] = True  # 标记为成功救援
                    # 设置结束时间（用于统计）
                    if "end_time" not in disasters[(target_x, target_y)]:
                        disasters[(target_x, target_y)]["end_time"] = time.time()
                    rescuer["target"] = None  # 任务完成，清除目标
    
    # 清除救援人员的无效目标
    for rescuer in rescuers:
        # 如果救援人员的目标是已完成救援或自然结束的灾情点，清除其目标
        if "target" in rescuer and rescuer["target"] is not None and rescuer["target"] in disasters:
            target_x, target_y = rescuer["target"]
            if disasters[(target_x, target_y)].get("frozen_rescue", False) or disasters[(target_x, target_y)].get("frozen_level", False):
                rescuer["target"] = None
