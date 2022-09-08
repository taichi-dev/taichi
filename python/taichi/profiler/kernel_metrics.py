from taichi._lib import core as _ti_core


class CuptiMetric:
    """A class to add CUPTI metric for :class:`~taichi.profiler.kernel_profiler.KernelProfiler`.

    This class is designed to add user selected CUPTI metrics.
    Only available for the CUDA backend now, i.e. you need ``ti.init(kernel_profiler=True, arch=ti.cuda)``.
    For usage of this class, see examples in func :func:`~taichi.profiler.set_kernel_profiler_metrics` and :func:`~taichi.profiler.collect_kernel_profiler_metrics`.

    Args:
        name (str): name of metric that collected by CUPTI toolkit. used by :func:`~taichi.profiler.set_kernel_profiler_metrics` and :func:`~taichi.profiler.collect_kernel_profiler_metrics`.
        header (str): column header of this metric, used by :func:`~taichi.profiler.print_kernel_profiler_info`.
        val_format (str): format for print metric value (and unit of this value), used by :func:`~taichi.profiler.print_kernel_profiler_info`.
        scale (float): scale of metric value, used by :func:`~taichi.profiler.print_kernel_profiler_info`.

    Example::

        >>> import taichi as ti

        >>> ti.init(kernel_profiler=True, arch=ti.cuda)
        >>> num_elements = 128*1024*1024

        >>> x = ti.field(ti.f32, shape=num_elements)
        >>> y = ti.field(ti.f32, shape=())
        >>> y[None] = 0

        >>> @ti.kernel
        >>> def reduction():
        >>>     for i in x:
        >>>         y[None] += x[i]

        >>> global_op_atom = ti.profiler.CuptiMetric(
        >>>     name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
        >>>     header=' global.atom ',
        >>>     val_format='    {:8.0f} ')

        >>> # add and set user defined metrics
        >>> profiling_metrics = ti.profiler.get_predefined_cupti_metrics('global_access') + [global_op_atom]
        >>> ti.profiler.set_kernel_profile_metrics(profiling_metrics)

        >>> for i in range(16):
        >>>     reduction()
        >>> ti.profiler.print_kernel_profiler_info('trace')

    Note:
        For details about using CUPTI in Taichi, please visit https://docs.taichi-lang.org/docs/profiler#advanced-mode.
    """
    def __init__(self,
                 name='',
                 header='unnamed_header',
                 val_format='     {:8.0f} ',
                 scale=1.0):
        self.name = name
        self.header = header
        self.val_format = val_format
        self.scale = scale


# Global Memory Metrics
dram_utilization = CuptiMetric(
    name='dram__throughput.avg.pct_of_peak_sustained_elapsed',
    header=' global.uti ',
    val_format='   {:6.2f} % ')

dram_bytes_sum = CuptiMetric(name='dram__bytes.sum',
                             header='  global.R&W ',
                             val_format='{:9.3f} MB ',
                             scale=1.0 / 1024 / 1024)

dram_bytes_throughput = CuptiMetric(name='dram__bytes.sum.per_second',
                                    header=' global.R&W/s ',
                                    val_format='{:8.3f} GB/s ',
                                    scale=1.0 / 1024 / 1024 / 1024)

dram_bytes_read = CuptiMetric(name='dram__bytes_read.sum',
                              header='   global.R ',
                              val_format='{:8.3f} MB ',
                              scale=1.0 / 1024 / 1024)

dram_read_throughput = CuptiMetric(name='dram__bytes_read.sum.per_second',
                                   header='   global.R/s ',
                                   val_format='{:8.3f} GB/s ',
                                   scale=1.0 / 1024 / 1024 / 1024)

dram_bytes_write = CuptiMetric(name='dram__bytes_write.sum',
                               header='   global.W ',
                               val_format='{:8.3f} MB ',
                               scale=1.0 / 1024 / 1024)

dram_write_throughput = CuptiMetric(name='dram__bytes_write.sum.per_second',
                                    header='   global.W/s ',
                                    val_format='{:8.3f} GB/s ',
                                    scale=1.0 / 1024 / 1024 / 1024)

# Shared Memory Metrics
shared_utilization = CuptiMetric(
    name=
    'l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed',
    header=' uti.shared ',
    val_format='   {:6.2f} % ')

shared_transactions_load = CuptiMetric(
    name='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    header=' shared.trans.W ',
    val_format='     {:10.0f} ')

shared_transactions_store = CuptiMetric(
    name='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    header=' shared.trans.R ',
    val_format='     {:10.0f} ')

shared_bank_conflicts_store = CuptiMetric(
    name='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    header=' bank.conflict.W ',
    val_format='      {:10.0f} ')

shared_bank_conflicts_load = CuptiMetric(
    name='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    header=' bank.conflict.R ',
    val_format='      {:10.0f} ')

# Atomic Metrics
global_op_atom = CuptiMetric(
    name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
    header=' global.atom ',
    val_format='    {:8.0f} ')

global_op_reduction = CuptiMetric(
    name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum',
    header=' global.red ',
    val_format='   {:8.0f} ')

# Hardware Utilization Metrics
sm_throughput = CuptiMetric(
    name='smsp__cycles_active.avg.pct_of_peak_sustained_elapsed',
    header=' core.uti ',
    val_format=' {:6.2f} % ')

dram_throughput = CuptiMetric(
    name='gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    header='  mem.uti ',
    val_format=' {:6.2f} % ')

l1tex_throughput = CuptiMetric(
    name='l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
    header='   L1.uti ',
    val_format=' {:6.2f} % ')

l2_throughput = CuptiMetric(
    name='lts__throughput.avg.pct_of_peak_sustained_elapsed',
    header='   L2.uti ',
    val_format=' {:6.2f} % ')

# Misc Metrics
l1_hit_rate = CuptiMetric(name='l1tex__t_sector_hit_rate.pct',
                          header='   L1.hit ',
                          val_format=' {:6.2f} % ')

l2_hit_rate = CuptiMetric(name='lts__t_sector_hit_rate.pct',
                          header='   L2.hit ',
                          val_format=' {:6.2f} % ')

achieved_occupancy = CuptiMetric(
    name='sm__warps_active.avg.pct_of_peak_sustained_active',
    header=' occupancy',
    val_format='   {:6.0f} ')

# metric suite: global load & store
global_access = [
    dram_bytes_sum,
    dram_bytes_throughput,
    dram_bytes_read,
    dram_read_throughput,
    dram_bytes_write,
    dram_write_throughput,
]

# metric suite: shared load & store
shared_access = [
    shared_transactions_load,
    shared_transactions_store,
    shared_bank_conflicts_store,
    shared_bank_conflicts_load,
]

# metric suite: atomic access
atomic_access = [
    global_op_atom,
    global_op_reduction,
]

# metric suite: cache hit rate
cache_hit_rate = [
    l1_hit_rate,
    l2_hit_rate,
]

# metric suite: device throughput
device_utilization = [
    sm_throughput,
    dram_throughput,
    shared_utilization,
    l1tex_throughput,
    l2_throughput,
]

# Predefined metrics suites
predefined_cupti_metrics = {
    'global_access': global_access,
    'shared_access': shared_access,
    'atomic_access': atomic_access,
    'cache_hit_rate': cache_hit_rate,
    'device_utilization': device_utilization,
}


def get_predefined_cupti_metrics(name=''):
    """Returns the specified cupti metric.

    Accepted arguments are 'global_access', 'shared_access', 'atomic_access',
    'cache_hit_rate', 'device_utilization'.

    Args:
        name (str): cupti metri name.
    """
    if name not in predefined_cupti_metrics:
        _ti_core.warn("Valid Taichi predefined metrics list (str):")
        for key in predefined_cupti_metrics:
            _ti_core.warn(f"    '{key}'")
        return None
    return predefined_cupti_metrics[name]


# Default metrics list
default_cupti_metrics = [dram_bytes_sum]
"""The metrics list, each is an instance of the :class:`~taichi.profiler.CuptiMetric`.
Default to `dram_bytes_sum`.
"""

__all__ = ['CuptiMetric', 'get_predefined_cupti_metrics']
