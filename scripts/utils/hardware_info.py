import cpuinfo
from collections import namedtuple

CPUInfo = namedtuple("CPUInfo", [
  "model_name",
  "clock_rate_GHz",
  "l1_cache_size_KiB",
  "l2_cache_size_KiB",
  "l3_cache_size_KiB"
])


def get_cpu_info() -> CPUInfo:
  """Returns a namedtuple with information about the CPU."""
  info = cpuinfo.get_cpu_info()
  model_name = info["brand_raw"]

  clock_rate_GHz = float(info.get("hz_actual", -1)) / 1e9

  l1_cache_size_KiB = int(info.get("l1_data_cache_size", 0)) // 1024
  l2_cache_size_KiB = int(info.get("l2_cache_size", 0)) // 1024
  l3_cache_size_KiB = int(info.get("l3_cache_size", 0)) // 1024

  return CPUInfo(
    model_name=model_name,
    clock_rate_GHz=clock_rate_GHz,
    l1_cache_size_KiB=l1_cache_size_KiB,
    l2_cache_size_KiB=l2_cache_size_KiB,
    l3_cache_size_KiB=l3_cache_size_KiB
  )
