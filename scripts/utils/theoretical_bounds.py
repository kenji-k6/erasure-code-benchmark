from utils.utils import AxType, XorecVersion
from typing import Callable
import pandas as pd




def get_theoretical_xor_encode_time(
  perf_df: pd.DataFrame,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical minimum encoding time in ms (single threaded)"""
  # Sanity check
  assert(len(perf_df.loc[perf_df["block_size_B"] == blk_size_B, "num_cycles"].values) == 1)

  num_cycles = perf_df.loc[perf_df["block_size_B"] == blk_size_B, "num_cycles"].values[0]
  block_xor_time_ms = (num_cycles / clk_rate_GHz) / 1e6

  num_calls = num_data_blks # number of calls to xorec_xor_blocks_<version>
  
  return block_xor_time_ms * num_calls


def get_theoretical_xor_encode_throughput(
  perf_df: pd.DataFrame,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical maximum encoding throughput in Gbps (single threaded)"""
  encode_time_ms = get_theoretical_xor_encode_time(
    perf_df,
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
  perf_df: pd.DataFrame,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical minimum decoding time in ms (single threaded)"""
  # Sanity check
  assert(len(perf_df.loc[perf_df["block_size_B"] == blk_size_B, "num_cycles"].values) == 1)
  
  num_cycles = perf_df.loc[perf_df["block_size_B"] == blk_size_B, "num_cycles"].values[0]
  block_xor_time_ms = (num_cycles / clk_rate_GHz) / 1e6

  xors_per_par_blk = num_data_blks/num_parity_blks # number of XORs per parity block

  num_calls = num_lost_data_blks * xors_per_par_blk # number of calls to xorec_xor_blocks_<version>
  
  return block_xor_time_ms * num_calls


def get_theoretical_xor_decode_throughput(
  perf_df: pd.DataFrame,
  clk_rate_GHz: float,
  blk_size_B: int,
  num_data_blks: int,
  num_parity_blks: int,
  num_lost_data_blks: int
) -> float:
  """Calculate the theoretical maximum decoding throughput in Gbps (single threaded)"""
  decode_time_ms = get_theoretical_xor_decode_time(
    perf_df,
    clk_rate_GHz,
    blk_size_B,
    num_data_blks,
    num_parity_blks,
    num_lost_data_blks
  )
  data_size_bits = blk_size_B * 8 * num_data_blks
  decode_throughput_Gbps = (data_size_bits / 1e6) / decode_time_ms
  return decode_throughput_Gbps


def get_theoretical_bound_func(ax: AxType) -> Callable[[pd.DataFrame, float, int, int, int, int], float]:
  if ax == AxType.ENCODE_T: return get_theoretical_xor_encode_time
  if ax == AxType.ENCODE_TP: return get_theoretical_xor_encode_throughput
  if ax == AxType.DECODE_T: return get_theoretical_xor_decode_time
  if ax == AxType.DECODE_TP: return get_theoretical_xor_decode_throughput
  raise ValueError(f"Unsupported AxType: {ax}")
    