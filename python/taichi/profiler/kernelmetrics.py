class CuptiMetric:
    '''
    '''
    def __init__(self, name='', header='', format='', scale=1.0):
        #cupti metric
        self.name = name  #(str): metric name for init CuptiToolkit
        #formating
        self.header = header  #(str): header for formatted printing
        self.format = format  #(str): format for print metric value and unit
        self.scale = scale  #(double): scale for metric value


# When declare a metric in this way:
#
#   dram_utilization = CuptiMetric(
#       'dram__throughput.avg.pct_of_peak_sustained_elapsed',
#       ' global.uti ',
#       '   {:6.2f} % ')
#
# Code will be auto formatted as:
#
#   dram_utilization = CuptiMetric(
#       'dram__throughput.avg.pct_of_peak_sustained_elapsed', ' global.uti ',
#       '   {:6.2f} % ')
#
# Which is not good for reading

########################## Global Memory Metrics ##########################

dram_utilization = CuptiMetric()
dram_utilization.name = 'dram__throughput.avg.pct_of_peak_sustained_elapsed'
dram_utilization.header = ' global.uti '
dram_utilization.format = '   {:6.2f} % '

dram_bytes_sum = CuptiMetric()
dram_bytes_sum.name = 'dram__bytes.sum'
dram_bytes_sum.header = '  global.r&w '
dram_bytes_sum.format = '{:9.3f} MB '
dram_bytes_sum.scale = 1.0 / 1024 / 1024

dram_bytes_throughput = CuptiMetric()
dram_bytes_throughput.name = 'dram__bytes.sum.per_second'
dram_bytes_throughput.header = ' global.r&w/s '
dram_bytes_throughput.format = '{:8.3f} GB/s '
dram_bytes_throughput.scale = 1.0 / 1024 / 1024 / 1024

dram_bytes_read = CuptiMetric()
dram_bytes_read.name = 'dram__bytes_read.sum'
dram_bytes_read.header = '   global.r '
dram_bytes_read.format = '{:8.3f} MB '
dram_bytes_read.scale = 1.0 / 1024 / 1024

dram_read_throughput = CuptiMetric()
dram_read_throughput.name = 'dram__bytes_read.sum.per_second'
dram_read_throughput.header = '   global.r/s '
dram_read_throughput.format = '{:8.3f} GB/s '
dram_read_throughput.scale = 1.0 / 1024 / 1024 / 1024

dram_bytes_write = CuptiMetric()
dram_bytes_write.name = 'dram__bytes_write.sum'
dram_bytes_write.header = '   global.w '
dram_bytes_write.format = '{:8.3f} MB '
dram_bytes_write.scale = 1.0 / 1024 / 1024

dram_write_throughput = CuptiMetric()
dram_write_throughput.name = 'dram__bytes_write.sum.per_second'
dram_write_throughput.header = '   global.w/s '
dram_write_throughput.format = '{:8.3f} GB/s '
dram_write_throughput.scale = 1.0 / 1024 / 1024 / 1024

########################## Shared Memory Metrics ##########################

shared_utilization = CuptiMetric()
shared_utilization.name = 'l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed'
shared_utilization.header = ' uti.shared '
shared_utilization.format = '   {:6.2f} % '

shared_transactions_load = CuptiMetric()
shared_transactions_load.name = 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'
shared_transactions_load.header = ' shared.trans.w '
shared_transactions_load.format = '     {:10.0f} '

shared_transactions_store = CuptiMetric()
shared_transactions_store.name = 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum'
shared_transactions_store.header = ' shared.trans.r '
shared_transactions_store.format = '     {:10.0f} '

shared_bank_conflicts_store = CuptiMetric()
shared_bank_conflicts_store.name = 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'
shared_bank_conflicts_store.header = ' bank.conflict.w '
shared_bank_conflicts_store.format = '      {:10.0f} '

shared_bank_conflicts_load = CuptiMetric()
shared_bank_conflicts_load.name = 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'
shared_bank_conflicts_load.header = ' bank.conflict.r '
shared_bank_conflicts_load.format = '      {:10.0f} '

########################## Atomic Metrics ##########################

global_op_atom = CuptiMetric()
global_op_atom.name = 'l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum'
global_op_atom.header = ' global.atom '
global_op_atom.format = '    {:8.0f} '

global_op_reduction = CuptiMetric()
global_op_reduction.name = 'l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum'
global_op_reduction.header = ' global.red '
global_op_reduction.format = '   {:8.0f} '

################# Hardware Utilization Metrics #####################

sm_throughput = CuptiMetric()
sm_throughput.name = 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
sm_throughput.header = ' core.uti '
sm_throughput.format = ' {:6.2f} % '

dram_throughput = CuptiMetric()
dram_throughput.name = 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'
dram_throughput.header = '  mem.uti '
dram_throughput.format = ' {:6.2f} % '

l1tex_throughput = CuptiMetric()
l1tex_throughput.name = 'l1tex__throughput.avg.pct_of_peak_sustained_elapsed'
l1tex_throughput.header = '   L1.uti '
l1tex_throughput.format = ' {:6.2f} % '

l2_throughput = CuptiMetric()
l2_throughput.name = 'lts__throughput.avg.pct_of_peak_sustained_elapsed'
l2_throughput.header = '   L2.uti '
l2_throughput.format = ' {:6.2f} % '

########################## Misc Metrics ##########################

l1_hit_rate = CuptiMetric()
l1_hit_rate.name = 'l1tex__t_sector_hit_rate.pct'
l1_hit_rate.header = '   L1.hit '
l1_hit_rate.format = ' {:6.2f} % '

l2_hit_rate = CuptiMetric()
l2_hit_rate.name = 'lts__t_sector_hit_rate.pct'
l2_hit_rate.header = '   L2.hit '
l2_hit_rate.format = ' {:6.2f} % '

achieved_occupancy = CuptiMetric()
achieved_occupancy.name = 'sm__warps_active.avg.pct_of_peak_sustained_active'
achieved_occupancy.header = ' occupancy'
achieved_occupancy.format = '   {:6.0f} '

####  global load & store ####
global_access_metrics = [
    dram_utilization,
    dram_bytes_sum,
    dram_bytes_throughput,
    dram_bytes_read,
    dram_read_throughput,
    dram_bytes_write,
    dram_write_throughput,
]

#### shared load & store ####
shared_access_metrics = [
    shared_utilization,
    shared_transactions_load,
    shared_transactions_store,
    shared_bank_conflicts_store,
    shared_bank_conflicts_load,
]

#### atomic access ####
atomic_access_metrics = [
    global_op_atom,
    global_op_reduction,
]

#### cache hit ####
cache_hit_metrics = [
    l1_hit_rate,
    l2_hit_rate,
]

#### device throughput ####
device_utilization_metrics = [
    sm_throughput,
    dram_throughput,
    shared_utilization,
    l1tex_throughput,
    l2_throughput,
]

# Default metrics list
default_metric_list = [dram_bytes_sum]