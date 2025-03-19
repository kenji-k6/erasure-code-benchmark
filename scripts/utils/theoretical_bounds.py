from utils.utils import AxType
from typing import Callable

def get_theoretical_xor_encode_time(
  simd_xor_latency: int,
  simd_xor_throughput: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical minimum encoding time in ms (single threaded)"""
  blk_size_bits = blk_size_B * 8
  num_xors_per_block = blk_size_bits / simd_bits
  num_cycles_per_block = simd_xor_latency + (num_xors_per_block - 1) * simd_xor_throughput
  block_xor_time_ms = (num_cycles_per_block / clk_rate_GHz) / 1e6
  return block_xor_time_ms * num_data_blks


def get_theoretical_xor_encode_throughput(
  simd_xor_latency: int,
  simd_xor_throughput: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical maximum encoding throughput in Gbps (single threaded)"""
  encode_time_ms = get_theoretical_xor_encode_time(
    simd_xor_latency,
    simd_xor_throughput,
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


def get_theoretical_xor_decode_time(
  simd_xor_latency: int,
  simd_xor_throughput: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical minimum decoding time in ms (single threaded)"""
  blk_size_bits = blk_size_B * 8
  num_xors_per_block = blk_size_bits / simd_bits
  num_cycles_per_block = simd_xor_latency + (num_xors_per_block - 1) * simd_xor_throughput
  block_xor_time_ms = (num_cycles_per_block / clk_rate_GHz) / 1e6

  blks_per_par_blk = num_data_blks / num_parity_blks

  time_per_lst_blk = block_xor_time_ms * blks_per_par_blk
  return time_per_lst_blk * num_lost_data_blks


def get_theoretical_xor_decode_throughput(
  simd_xor_latency: int,
  simd_xor_throughput: float,
  simd_bits: int,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical maximum decoding throughput in Gbps (single threaded)"""
  decode_time_ms = get_theoretical_xor_decode_time(
    simd_xor_latency,
    simd_xor_throughput,
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




def get_theoretical_bound_func(ax: AxType) -> Callable[[int, float, int, float, int, int, int, int], float]:
  if ax == AxType.ENCODE_T: return get_theoretical_xor_encode_time
  if ax == AxType.ENCODE_TP: return get_theoretical_xor_encode_throughput
  if ax == AxType.DECODE_T: return get_theoretical_xor_decode_time
  if ax == AxType.DECODE_TP: return get_theoretical_xor_decode_throughput
  raise ValueError(f"Unsupported AxType: {ax}")
    