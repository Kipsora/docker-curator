import tensorflow

__all__ = ['get_devices', 'get_cpu_nums', 'get_gpu_nums']


def get_devices(device_type):
    return tensorflow.config.experimental.list_physical_devices(device_type)


def get_gpu_nums():
    return len(get_devices("GPU"))


def get_cpu_nums():
    return len(get_devices("CPU"))
