# 全局参数配置

GRID_SIZE = 20  # 城市网格大小
NUM_RESCUERS = 5  # 初始救援人员数量
DISASTER_SPAWN_RATE = 0.2  # 每个时间步新增灾情点的概率
CRITICAL_DISASTER_THRESHOLD = 8  # 重大灾情等级
RESOURCE_SATURATION_THRESHOLD = 1.0  # 资源饱和度阈值

# 任务参数
MAX_RESCUE_CAPACITY = 3  # 每个救援人员的最大救援能力
MAX_SPEED = 2  # 救援人员最大移动速度
SIMULATION_TIME = 200  # 总模拟时间

# 统计参数
STATS_WINDOW_SIZE = 20  # 统计窗口大小（用于计算救援成功率）
