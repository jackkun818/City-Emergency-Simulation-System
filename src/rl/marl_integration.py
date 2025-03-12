from core import config
from .marl_rescue import MARLController
import os

# 全局MARL控制器实例
marl_controller = None

def initialize_marl_controller(env):
    """初始化MARL控制器"""
    global marl_controller
    
    # 如果已经初始化，则直接返回
    if marl_controller is not None:
        return marl_controller
    
    # 创建MARL控制器
    marl_controller = MARLController(
        grid_size=env.GRID_SIZE,
        num_rescuers=len(env.rescuers),
        hidden_dim=config.MARL_CONFIG["hidden_dim"],
        lr=config.MARL_CONFIG["learning_rate"],
        gamma=config.MARL_CONFIG["gamma"]
    )
    
    # 尝试加载已训练的模型
    model_path = config.MARL_CONFIG["model_save_path"]
    if os.path.exists(model_path):
        print(f"加载MARL模型: {model_path}")
        marl_controller.load_models(model_path)
    else:
        print(f"未找到MARL模型，将使用随机策略")
    
    return marl_controller


def marl_rescue_dispatch(rescuers, disasters, grid_size, current_time_step=0):
    """
    使用MARL控制器分配救援任务
    
    参数:
    - rescuers: 救援人员列表
    - disasters: 灾情点字典
    - grid_size: 网格大小
    - current_time_step: 当前时间步
    """
    # 创建临时环境对象用于MARL
    class TempEnv:
        def __init__(self, rescuers, disasters, grid_size):
            self.rescuers = rescuers
            self.disasters = disasters
            self.GRID_SIZE = grid_size
    
    temp_env = TempEnv(rescuers, disasters, grid_size)
    
    # 初始化或获取MARL控制器
    controller = initialize_marl_controller(temp_env)
    
    # 获取MARL推荐的动作
    actions = controller.select_actions(rescuers, disasters, training=False)
    
    # 将动作转换为任务分配
    for i, action in enumerate(actions):
        if i >= len(rescuers):
            break
            
        # 跳过动作为0的情况（不移动）
        if action == 0:
            continue
            
        # 计算目标位置
        target_pos = controller.action_to_target(action, grid_size)
        
        # 只有目标位置有灾情时才分配任务
        if target_pos in disasters:
            rescuers[i]["target"] = target_pos


def hybrid_marl_rescue_dispatch(rescuers, disasters, grid_size, current_time_step=0):
    """
    混合策略：结合传统算法和MARL进行任务分配
    
    参数:
    - rescuers: 救援人员列表
    - disasters: 灾情点字典
    - grid_size: 网格大小
    - current_time_step: 当前时间步
    """
    from core.rescue_dispatch import hybrid_rescue_dispatch
    
    # 将救援人员分为两组
    num_rescuers = len(rescuers)
    marl_rescuers_count = num_rescuers // 2  # 使用MARL的救援人员数量
    
    # MARL组
    marl_rescuers = rescuers[:marl_rescuers_count]
    # 传统算法组
    traditional_rescuers = rescuers[marl_rescuers_count:]
    
    # 使用MARL为第一组分配任务
    marl_rescue_dispatch(marl_rescuers, disasters, grid_size, current_time_step)
    
    # 使用传统算法为第二组分配任务
    hybrid_rescue_dispatch(traditional_rescuers, disasters, grid_size)


def get_algorithm_name():
    """根据配置返回当前使用的算法名称"""
    algorithm_names = {
        0: "传统混合算法",
        1: "多智能体强化学习",
        2: "混合算法"
    }
    return algorithm_names.get(config.TASK_ALLOCATION_ALGORITHM, "未知算法")


def dispatch_rescue_tasks(rescuers, disasters, grid_size, current_time_step=0):
    """
    根据配置选择使用哪种任务分配算法
    
    参数:
    - rescuers: 救援人员列表
    - disasters: 灾情点字典
    - grid_size: 网格大小
    - current_time_step: 当前时间步
    """
    algorithm = config.TASK_ALLOCATION_ALGORITHM
    
    if algorithm == 0:
        # 使用传统混合算法
        from core.rescue_dispatch import hybrid_rescue_dispatch
        hybrid_rescue_dispatch(rescuers, disasters, grid_size)
    elif algorithm == 1:
        # 使用多智能体强化学习
        marl_rescue_dispatch(rescuers, disasters, grid_size, current_time_step)
    elif algorithm == 2:
        # 使用混合算法
        hybrid_marl_rescue_dispatch(rescuers, disasters, grid_size, current_time_step)
    else:
        # 默认使用传统混合算法
        from core.rescue_dispatch import hybrid_rescue_dispatch
        hybrid_rescue_dispatch(rescuers, disasters, grid_size) 