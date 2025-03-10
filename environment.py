import numpy as np#1
import random
import time
import random
# 定义城市网格大小
GRID_SIZE = 10  # 10x10 的网格地图
NUM_DISASTERS = 5  # 初始灾情点数量
NUM_RESCUERS = 3  # 初始救援人员数量


class Environment:
    def __init__(self, grid_size, num_rescuers):
        self.GRID_SIZE = grid_size
        self.num_rescuers = num_rescuers
        self.rescuers = []
        self.disasters = {}

        self.initialize_rescuers()

    def initialize_rescuers(self):
        """ 初始化救援人员并设置差异化的能力和速度 """
        for i in range(self.num_rescuers):
            # 随机生成每个救援人员的能力和速度
            capacity = random.randint(1, 3)  # 救援能力在1到3之间
            speed = random.randint(1, 3)  # 移动速度在1到3之间

            self.rescuers.append({
                "id": i,
                "position": (0, 0),  # 初始位置可以是任意的
                "active_time": 0,
                "capacity": capacity,
                "speed": speed
            })

    def update_disasters(self, current_time_step=None):
        """ 
        模拟灾情的出现和变化
        :param current_time_step: 当前的时间步，用于记录灾情点的创建时间
        """
        # 更新红叉计数器
        for pos, disaster in list(self.disasters.items()):
            # 如果红叉计数器大于0，则递减
            if disaster.get("show_red_x", 0) > 0:
                disaster["show_red_x"] -= 1
                if disaster["show_red_x"] == 0:
                    print(f"📍 灾情点 {pos} 红叉显示时间结束，不再显示")

        # 随机生成新的灾情点
        new_disaster_chance = 0.2
        for _ in range(int(new_disaster_chance * self.GRID_SIZE)):  # 减少生成频率，避免地图过于拥挤
            x, y = np.random.randint(0, self.GRID_SIZE, size=2)
            if (x, y) not in self.disasters:
                # 先生成level，范围5-10
                level = np.random.randint(5, 11)  # 注意上限改为11，使范围包含10
                
       
                if level <= 6:
                    rescue_needed = np.random.randint(5, 6)  
                elif level <= 8:
                    rescue_needed = np.random.randint(7, 8)  
                else:
                    rescue_needed = np.random.randint(9, 10)  
                
                # 新灾情点加入初始时间和时间步信息
                self.disasters[(x, y)] = {
                    "level": level,
                    "rescue_needed": rescue_needed,
                    "start_time": time.time(),  # 记录灾情点出现的时间
                    "time_step": current_time_step,  # 记录灾情点创建的时间步
                    "frozen_level": False,  # 初始状态为未冻结
                    "frozen_rescue": False,  # 初始状态为未冻结
                    "rescue_success": False,  # 初始状态为未救援成功
                    "show_red_x": 0  # 红叉显示计数器，0表示不显示
                }
                print(
                    f"🔴 新灾情点出现在 {x, y}，等级：{self.disasters[(x, y)]['level']}，需要救援：{self.disasters[(x, y)]['rescue_needed']}")

        # 自然减弱已有灾情（灾情会随时间自然减弱）
        for pos, disaster in list(self.disasters.items()):  # 使用list复制，避免在迭代中修改字典
            # 只跳过rescue_needed=0的灾情点，不再跳过level=0的点
            if disaster.get("frozen_rescue", False):
                continue

            if disaster["level"] > 0:
                disaster["level"] -= np.random.randint(0, 2)  # 随机减弱0-1点
                disaster["level"] = max(0, disaster["level"])  # 确保不会为负

                # 如果自然减弱导致level降至0但rescue_needed>0，标记为救援失败
                if disaster["level"] <= 0 and disaster.get("rescue_needed", 0) > 0:
                    print(f"⚠️ 灾情点 {pos} 自然减弱至level=0但仍需救援，标记为救援失败！")
                    disaster["show_red_x"] = 2  # 显示红叉
                    disaster["frozen_level"] = True  # 冻结level，防止进一步减弱
                    disaster["rescue_success"] = False  # 明确标记为救援失败
                    # 设置结束时间步
                    if current_time_step:
                        disaster["end_time"] = current_time_step
                    print(f"⚪ 灾情点 {pos} 未能成功救援！")


if __name__ == "__main__":
    env = Environment(GRID_SIZE, NUM_RESCUERS)  # Create environment instance
    print("City Map:")  # 输出城市地图
    print(env.disasters)  # 输出当前灾情信息

    print("\nDisaster Points:")  # 输出灾情点信息
    for key, value in env.disasters.items():
        print(f"Location {key}, Level: {value['level']}, Rescue Needed: {value['rescue_needed']}")  # 输出每个灾情点的具体信息

    print("\nRescuer Information:")  # 输出救援人员信息
    for rescuer in env.rescuers:
        print(
            f"ID {rescuer['id']}, Position: {rescuer['position']}, Speed: {rescuer['speed']}, Capacity: {rescuer['capacity']}")  # 输出救援人员编号、位置、速度和救援能力
