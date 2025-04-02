#include "abstract_bm.hpp"
#include <omp.h>

ECBenchmark::ECBenchmark(const BenchmarkConfig& config) noexcept
    : m_size_msg(config.message_size),
      m_size_blk(config.block_size),
      m_fec_params(config.fec_params),
      m_num_lst_rdma_pkts(config.num_lost_rmda_packets),
      m_num_threads(config.num_cpu_threads)
  {
    m_size_data_submsg = get<0>(config.fec_params)*config.block_size;
    m_size_parity_submsg = get<1>(config.fec_params)*config.block_size;

    m_num_chunks = config.message_size/m_size_data_submsg;
    m_blks_per_chunk = get<0>(config.fec_params)+get<1>(config.fec_params);
    m_data_blks_per_chunk = get<0>(config.fec_params);
    m_parity_blks_per_chunk = get<1>(config.fec_params);
    omp_set_num_threads(config.num_cpu_threads);
  }

  bool ECBenchmark::check_for_corruption() const noexcept {
    for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
      if (!validate_block(m_data_buffer+i*m_size_blk, m_size_blk)) return false;
    }
    return true;
  }