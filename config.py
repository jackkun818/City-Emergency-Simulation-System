# 全局参数配置

# 灾难规模选择（小型=0, 中型=1, 大型=2, 自定义=3）
DISASTER_SCALE = 1  # 默认使用中型灾难


# 灾难规模预设
DISASTER_PRESETS = {
    # 小型灾难
    0: {
        "name": "小型灾难",
        "grid_size": 10,                 # 小型网格
        "disaster_spawn_rate": 0.1,      # 较低的灾情生成概率
        "enable_spawn_rate_decay": True, # 启用灾情概率衰减
        "initial_spawn_rate_multiplier": 1.5, # 初始灾情概率倍数（较小）
        "spawn_rate_decay_steps": 80     # 灾情概率衰减的时间步数（较短）
    },
    # 中型灾难
    1: {
        "name": "中型灾难",
        "grid_size": 20,                 # 中型网格
        "disaster_spawn_rate": 0.15,      # 中等的灾情生成概率
        "enable_spawn_rate_decay": True, # 启用灾情概率衰减
        "initial_spawn_rate_multiplier": 2.0, # 初始灾情概率倍数（中等）
        "spawn_rate_decay_steps": 100    # 灾情概率衰减的时间步数（中等）
    },
    # 大型灾难
    2: {
        "name": "大型灾难",
        "grid_size": 30,                 # 大型网格
        "disaster_spawn_rate": 0.2,      # 较高的灾情生成概率
        "enable_spawn_rate_decay": True, # 启用灾情概率衰减
        "initial_spawn_rate_multiplier": 2.5, # 初始灾情概率倍数（较大）
        "spawn_rate_decay_steps": 120    # 灾情概率衰减的时间步数（较长）
    }
}

# 以下参数当DISASTER_SCALE=3（自定义）时使用
GRID_SIZE = 20  # 城市网格大小
DISASTER_SPAWN_RATE = 0.2  # 每个时间步新增灾情点的概率
ENABLE_SPAWN_RATE_DECAY = True  # 是否启用灾情概率衰减
INITIAL_SPAWN_RATE_MULTIPLIER = 2.0  # 初始灾情概率倍数
SPAWN_RATE_DECAY_STEPS = 100  # 灾情概率衰减的时间步数


# 其他全局参数
CRITICAL_DISASTER_THRESHOLD = 8  # 重大灾情等级
RESOURCE_SATURATION_THRESHOLD = 1.0  # 资源饱和度阈值
NUM_RESCUERS = 10  # 默认救援人员数量
MAX_RESCUE_CAPACITY = 3  # 每个救援人员的最大救援能力
MAX_SPEED = 2  # 救援人员最大移动速度
SIMULATION_TIME = 500  # 总模拟时间
STATS_WINDOW_SIZE = 30  # 统计窗口大小（用于计算救援成功率）

# 根据灾难规模获取实际的配置参数
def get_config_param(param_name):
    """根据当前的灾难规模获取对应的配置参数值"""
    if DISASTER_SCALE == 3:  # 自定义模式
        if param_name == "grid_size":
            return GRID_SIZE
        elif param_name == "disaster_spawn_rate":
            return DISASTER_SPAWN_RATE
        elif param_name == "enable_spawn_rate_decay":
            return ENABLE_SPAWN_RATE_DECAY
        elif param_name == "initial_spawn_rate_multiplier":
            return INITIAL_SPAWN_RATE_MULTIPLIER
        elif param_name == "spawn_rate_decay_steps":
            return SPAWN_RATE_DECAY_STEPS
        elif param_name == "num_rescuers":  # 救援人员数量独立于灾难规模
            return NUM_RESCUERS
    else:  # 预设模式
        if param_name == "num_rescuers":  # 救援人员数量独立于灾难规模
            return NUM_RESCUERS
        # 其他参数从预设中获取
        return DISASTER_PRESETS[DISASTER_SCALE].get(param_name)
        
# 计算当前时间步的实际灾情生成概率
def get_actual_spawn_rate(time_step):
    """根据当前时间步计算实际的灾情生成概率，考虑初始翻倍和衰减"""
    base_rate = get_config_param("disaster_spawn_rate")
    enable_decay = get_config_param("enable_spawn_rate_decay")
    
    if not enable_decay:
        return base_rate
    
    # 获取衰减参数
    initial_multiplier = get_config_param("initial_spawn_rate_multiplier")
    decay_steps = get_config_param("spawn_rate_decay_steps")
    
    # 计算衰减系数（从initial_multiplier线性衰减到1.0）
    if time_step >= decay_steps:
        decay_factor = 1.0
    else:
        decay_factor = initial_multiplier - (initial_multiplier - 1.0) * (time_step / decay_steps)
    
    return base_rate * decay_factor
