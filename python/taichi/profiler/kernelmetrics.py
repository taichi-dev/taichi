from taichi.core import ti_core as _ti_core


class CuptiMetric:
    """A class to add CUPTI metric for :class:`~taichi.lang.KernelProfiler`.

    This class is designed to add user selected CUPTI metrics.
    Only available for the CUDA backend now, i.e. you need ``ti.init(kernel_profiler=True, arch=ti.cuda)``.
    For usage of this class, see examples in func :func:`~taichi.lang.set_kernel_profile_metrics` and :func:`~taichi.lang.collect_kernel_profile_metrics`.

    Args:
        name (str): name of metric that collected by CUPTI toolkit. used by :func:`~taichi.lang.set_kernel_profile_metrics` and :func:`~taichi.lang.collect_kernel_profile_metrics`.
        header (str): column header of this metric, used by :func:`~taichi.lang.print_kernel_profile_info`.
        format (str): format for print metric value (and unit of this value), used by :func:`~taichi.lang.print_kernel_profile_info`.
        scale (float): scale of metric value, used by :func:`~taichi.lang.print_kernel_profile_info`.

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

        >>> global_op_atom = ti.CuptiMetric(
        >>>     name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
        >>>     header=' global.atom ',
        >>>     format='    {:8.0f} ')

        >>> # add and set user defined metrics
        >>> profiling_metrics = ti.get_predefined_cupti_metrics('global_access') + [global_op_atom]
        >>> ti.set_kernel_profile_metrics(profiling_metrics)

        >>> for i in range(16):
        >>>     reduction()
        >>> ti.print_kernel_profile_info('trace')

    Note:
        For details about using CUPTI in Taichi, please visit https://docs.taichi.graphics/docs/lang/articles/misc/profiler#advanced-mode.
    """
    def __init__(self,
                 name='',
                 header='unnamed_header',
                 format='     {:8.0f} ',
                 scale=1.0):
        self.name = name
        self.header = header
        self.format = format
        self.scale = scale


# Global Memory Metrics
dram_utilization = CuptiMetric(
    name='dram__throughput.avg.pct_of_peak_sustained_elapsed',
    header=' global.uti ',
    format='   {:6.2f} % ')

dram_bytes_sum = CuptiMetric(name='dram__bytes.sum',
                             header='  global.R&W ',
                             format='{:9.3f} MB ',
                             scale=1.0 / 1024 / 1024)

dram_bytes_throughput = CuptiMetric(name='dram__bytes.sum.per_second',
                                    header=' global.R&W/s ',
                                    format='{:8.3f} GB/s ',
                                    scale=1.0 / 1024 / 1024 / 1024)

dram_bytes_read = CuptiMetric(name='dram__bytes_read.sum',
                              header='   global.R ',
                              format='{:8.3f} MB ',
                              scale=1.0 / 1024 / 1024)

dram_read_throughput = CuptiMetric(name='dram__bytes_read.sum.per_second',
                                   header='   global.R/s ',
                                   format='{:8.3f} GB/s ',
                                   scale=1.0 / 1024 / 1024 / 1024)

dram_bytes_write = CuptiMetric(name='dram__bytes_write.sum',
                               header='   global.W ',
                               format='{:8.3f} MB ',
                               scale=1.0 / 1024 / 1024)

dram_write_throughput = CuptiMetric(name='dram__bytes_write.sum.per_second',
                                    header='   global.W/s ',
                                    format='{:8.3f} GB/s ',
                                    scale=1.0 / 1024 / 1024 / 1024)

# Shared Memory Metrics
shared_utilization = CuptiMetric(
    name=
    'l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed',
    header=' uti.shared ',
    format='   {:6.2f} % ')

shared_transactions_load = CuptiMetric(
    name='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    header=' shared.trans.W ',
    format='     {:10.0f} ')

shared_transactions_store = CuptiMetric(
    name='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    header=' shared.trans.R ',
    format='     {:10.0f} ')

shared_bank_conflicts_store = CuptiMetric(
    name='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    header=' bank.conflict.W ',
    format='      {:10.0f} ')

shared_bank_conflicts_load = CuptiMetric(
    name='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    header=' bank.conflict.R ',
    format='      {:10.0f} ')

# Atomic Metrics
global_op_atom = CuptiMetric(
    name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
    header=' global.atom ',
    format='    {:8.0f} ')

global_op_reduction = CuptiMetric(
    name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum',
    header=' global.red ',
    format='   {:8.0f} ')

# Hardware Utilization Metrics
sm_throughput = CuptiMetric(
    name='sm__throughput.avg.pct_of_peak_sustained_elapsed',
    header=' core.uti ',
    format=' {:6.2f} % ')

dram_throughput = CuptiMetric(
    name='gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    header='  mem.uti ',
    format=' {:6.2f} % ')

l1tex_throughput = CuptiMetric(
    name='l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
    header='   L1.uti ',
    format=' {:6.2f} % ')

l2_throughput = CuptiMetric(
    name='lts__throughput.avg.pct_of_peak_sustained_elapsed',
    header='   L2.uti ',
    format=' {:6.2f} % ')

# Misc Metrics
l1_hit_rate = CuptiMetric(name='l1tex__t_sector_hit_rate.pct',
                          header='   L1.hit ',
                          format=' {:6.2f} % ')

l2_hit_rate = CuptiMetric(name='lts__t_sector_hit_rate.pct',
                          header='   L2.hit ',
                          format=' {:6.2f} % ')

achieved_occupancy = CuptiMetric(
    name='sm__warps_active.avg.pct_of_peak_sustained_active',
    header=' occupancy',
    format='   {:6.0f} ')

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
    if name not in predefined_cupti_metrics:
        _ti_core.warn("Valid Taichi predefined metrics list (str):")
        for key in predefined_cupti_metrics:
            _ti_core.warn(f"    '{key}'")
        return None
    else:
        return predefined_cupti_metrics[name]


# Default metrics list
default_cupti_metrics = [dram_bytes_sum]
