import ray

ray_gpu_num = 0.3

class ParallelConfig():
    def __init__(self, ray_flag):
        ray.shutdown()
        if ray_flag:
            ray.init()  # 并行模式
        else:
            ray.init(local_mode=True)  # 串行模式