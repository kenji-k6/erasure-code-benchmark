from utils.utils import AxType
from typing import Callable

def get_theoretical_encode_time(
  simd_xor_cpi: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical minimum encoding time in ms (single threaded)"""
  blk_size_bits = blk_size_B * 8
  num_xors = (num_data_blks * blk_size_bits) / simd_bits
  num_cycles = num_xors * simd_xor_cpi
  encode_time_ms = (num_cycles / clk_rate_GHz) / 1e6 # equal to (#cycles / (#GHz * 1e9)) * 1e3  = (#cycles / #Hz) * 1e3
  return encode_time_ms

def get_theoretical_encode_throughput(
  simd_xor_cpi: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical maximum encoding throughput in Gbps (single threaded)"""
  encode_time_ms = get_theoretical_encode_time(
    simd_xor_cpi,
    simd_bits,
    clk_rate_GHz,
    blk_size_B,
    num_data_blks,
    num_parity_blks,
    num_lost_data_blks
  )
  data_size_bits = blk_size_B * 8 * num_data_blks
  encode_throughput_Gbps = (data_size_bits / 1e6) / encode_time_ms # equal to (#bits/1e9)/(#ms/1e3) = #Gbps
  return encode_throughput_Gbps


def get_theoretical_decode_time(
  simd_xor_cpi: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical minimum decoding time in ms (single threaded)"""
  blk_size_bits = blk_size_B * 8
  blks_per_par_blk = num_data_blks / num_parity_blks
  num_xors_per_lst_blk = (blks_per_par_blk * blk_size_bits) / simd_bits
  num_xors = num_xors_per_lst_blk * num_lost_data_blks
  num_cycles = num_xors * simd_xor_cpi
  decode_time_ms = (num_cycles / clk_rate_GHz) / 1e6
  return decode_time_ms


def get_theoretical_decode_throughput(
  simd_xor_cpi: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical maximum decoding throughput in Gbps (single threaded)"""
  decode_time_ms = get_theoretical_decode_time(
    simd_xor_cpi,
    simd_bits,
    clk_rate_GHz,
    blk_size_B,
    num_data_blks,
    num_parity_blks,
    num_lost_data_blks
  )
  data_size_bits = blk_size_B * 8 * num_data_blks
  decode_throughput_Gbps = (data_size_bits / 1e6) / decode_time_ms
  return decode_throughput_Gbps

def get_theoretical_bound_func(ax: AxType) -> Callable[[float, int, float, int, int, int, int], float]:
  if ax == AxType.ENCODE_T: return get_theoretical_encode_time
  if ax == AxType.ENCODE_TP: return get_theoretical_encode_throughput
  if ax == AxType.DECODE_T: return get_theoretical_decode_time
  if ax == AxType.DECODE_TP: return get_theoretical_decode_throughput
  raise ValueError(f"Unsupported AxType: {ax}")
    