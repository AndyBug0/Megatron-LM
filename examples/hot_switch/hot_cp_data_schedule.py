import copy
import enum
import math
import sys
import time
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.data_schedule import BaseScheduler

from pipeline_simulator.simulator.schedules import SplitFuseSchedule, InterleavedSchedule
from pipeline_simulator.simulator.solver import test_with_schedule


def compute_pp_bubble_ratio(PP, m, VPP=1):
    return (PP - 1) / (m * VPP + PP - 1)


def greedy_assign_bucket_to_dp(curr_m, indices_buckets, normal_indexes, except_buckets, except_bucket_num_per_sample, 
                             except_bucket_m_per_sample, except_bucket_dp_per_sample, buckets_for_current_m,
                             dp_size_for_current_m, used_flops, used_fwd_flops, used_bwd_flops, bucket_num_per_dp_curr_m,
                             all_flops, all_lengths, combination=None, config=None):
    """
    使用贪心算法将桶分配给数据并行(DP)组
    
    参数:
        curr_m: 当前处理的m值(微批次数量)
        indices_buckets: 所有桶的索引信息
        except_buckets: 特殊处理的序列桶
        except_bucket_num_per_sample: 每个特殊序列分配的桶数
        except_bucket_m_per_sample: 每个特殊序列分配的m值
        buckets_for_current_m: 当前m值对应的桶列表
        dp_size_for_current_m: 当前m值的DP组大小
        used_flops: 已使用的总FLOPs
        used_fwd_flops: 已使用的前向FLOPs
        used_bwd_flops: 已使用的后向FLOPs
        bucket_num_per_dp_curr_m: 每个DP rank的桶数量限制
        
    返回:
        包含以下内容的元组:
        - 每个DP rank的总FLOPs列表
        - 每个DP rank的前向FLOPs列表
        - 每个DP rank的后向FLOPs列表
        - 分配给每个DP rank的桶列表
        - 是否遇到空桶的标志
    """

    # args = get_args()
    
    # 初始化每个DP rank的统计列表
    fwd_flops_for_dp_per_m = [[] for _ in range(dp_size_for_current_m)]  # 前向FLOPs
    bwd_flops_for_dp_per_m = [[] for _ in range(dp_size_for_current_m)]  # 后向FLOPs
    seq_len_for_dp_per_m = [[] for _ in range(dp_size_for_current_m)]  # 每个 microbatch 的 seqlen
    buckets_for_dp = [[] for _ in range(dp_size_for_current_m)]  # 分配的桶
    sample_ids_for_dp = [[] for _ in range(dp_size_for_current_m)]  # 分配的 sample_id
    sample_lengths_for_dp = [[] for _ in range(dp_size_for_current_m)]  # 分配的 sample_length

    # 初始化每个DP rank的FLOPs总和和已用桶数
    fwd_flops_sum_per_dp_this_m = [0.0] * dp_size_for_current_m
    bucket_used_num_per_dp_this_m = [0] * dp_size_for_current_m
    prefix_sum_per_dp_this_m = 0  # 用于跟踪特殊序列桶的分配位置
    # 第一步：分配特殊序列(Seq1F1B)的桶
    # 遍历每个样本，判断其是否被分配到当前 m
    num_split_for_dp = [0] * dp_size_for_current_m
    # print_rank0(f"assign except bucket")
    for idx in range(len(except_bucket_m_per_sample)):
        # 只处理当前m值的特殊序列
        if (except_bucket_m_per_sample[idx] != curr_m):
            continue

        # 计算当前序列在except_buckets中的位置范围
        st = prefix_sum_per_dp_this_m
        ed = prefix_sum_per_dp_this_m + except_bucket_num_per_sample[idx]

        # 将序列片段分配到选定的DP rank
        for k in range(st, ed):
            # 记录FLOPs信息
            bucket_tmp = except_buckets[curr_m][k]

            bucket_tmp.fwd_flops = (bucket_tmp.fwd_flops, {}, bucket_tmp.cp_size, bucket_tmp.dp_index)
            bucket_tmp.bwd_flops = (bucket_tmp.bwd_flops, {}, bucket_tmp.cp_size, bucket_tmp.dp_index)

            fwd_flops_for_dp_per_m[bucket_tmp.dp_index].append(bucket_tmp.fwd_flops)
            bwd_flops_for_dp_per_m[bucket_tmp.dp_index].append(bucket_tmp.bwd_flops)
            
            # construct 2d array
            # correction for memory simulator
            seq_len_for_dp_per_m[bucket_tmp.dp_index].append([bucket_tmp.seq_len_sum // bucket_tmp.cp_size // config.min_hybrid_context_parallel_size * config.context_parallel_size])
            
            # 更新DP rank的负载统计
            fwd_flops_sum_per_dp_this_m[bucket_tmp.dp_index] += bucket_tmp.fwd_flops[0]

            buckets_for_dp[bucket_tmp.dp_index].append(bucket_tmp)
            sample_ids_for_dp[bucket_tmp.dp_index].append(bucket_tmp.samples)

            # 更新分配位置和桶使用计数
            num_split_for_dp[bucket_tmp.dp_index] += 1
            bucket_used_num_per_dp_this_m[bucket_tmp.dp_index] += 1
        prefix_sum_per_dp_this_m += except_bucket_num_per_sample[idx]
        # for ttt in bucket_used_num_per_dp_this_m:
        #     print_rank0(ttt, end='\t')
        # print_rank0("")

    # 第二步：分配普通桶
    # print_rank0(f"assign normal bucket")
    empty_bucket_flag = False
    for j in range(len(buckets_for_current_m)):
        # 寻找最适合的DP rank(负载最小且桶未满)
        min_flops = sys.float_info.max
        min_flops_dp_rank = -1
        for dp_rank in range(len(fwd_flops_sum_per_dp_this_m)):
            if (min_flops > fwd_flops_sum_per_dp_this_m[dp_rank]) and \
               (bucket_used_num_per_dp_this_m[dp_rank] < bucket_num_per_dp_curr_m):
                min_flops = fwd_flops_sum_per_dp_this_m[dp_rank]
                min_flops_dp_rank = dp_rank

        assert min_flops_dp_rank != -1  # 确保找到合适的DP rank

        # 获取当前桶ID并检查是否为空
        bucket_id = buckets_for_current_m[j][1]
        if not indices_buckets[bucket_id] or len(indices_buckets[bucket_id].samples) == 0:
            # for idx, except_b in enumerate(except_buckets[curr_m]):
            #     print_rank0(f"{idx=}, {except_b=}")
            #     print_rank0(except_b)
            # for idx, normal_b in enumerate(indices_buckets):
            #     print_rank0(f"{idx=}, {normal_b=}")
            #     print_rank0(normal_b)
            # import pdb; pdb.set_trace()
            empty_bucket_flag = True
        
        # for test only
        indices_buckets[bucket_id].samples_fwd_flops = [all_flops[1][indice] for indice in indices_buckets[bucket_id].samples]

        # tflops to time
        scale = 0.5
        length_sum = 0
        length_square_sum = 0
        # attn_fwd_tflops_sum = 0
        # gemm_fwd_tflops_sum = 0
        lengths = []
        # NOTE shenglong 
        hidden_size = config.hidden_size
        # hidden_size = 4096

        bucket_tmp = indices_buckets[bucket_id]
        for sample_id in bucket_tmp.samples:
            length = all_lengths[sample_id]
            # attn_fwd_tflops = attention_tflops(length, hidden_size, scale)
            # gemm_fwd_tflops = linear_tflops(length, config.hidden_size)

            length_sum += length
            lengths.append(length)
            length_square_sum += (length ** 2)
            # attn_fwd_tflops_sum += attn_fwd_tflops
            # gemm_fwd_tflops_sum += gemm_fwd_tflops

        # fwd_time, bwd_time, fwd_time_dict = flops_to_times(length_sum, length_square_sum, attn_fwd_tflops_sum)
        fwd_time, bwd_time = bucket_tmp.fwd_flops, bucket_tmp.bwd_flops     # TODO(wuguohao)
        fwd_time_dict = {}

        split_num = 1
        split_idx = 0
        bwd_time_dict = {} # TODO
        bucket_tmp.fwd_flops = (bucket_tmp.fwd_flops, fwd_time_dict, split_num, split_idx)
        bucket_tmp.bwd_flops = (bucket_tmp.bwd_flops, bwd_time_dict, split_num, split_idx)
        # bucket_tmp.fwd_flops = (bucket_tmp.fwd_flops, {"attn_fwd_time":attn_fwd_tflops_sum, "mlp_fc1_fwd_time":gemm_fwd_tflops_sum})

        # print_rank0(f"{lengths=}")
        assert length_sum == bucket_tmp.seq_len_sum, f"{length_sum=}, {bucket_tmp.seq_len_sum=}, {bucket_id=}"

        # !将带 offset 的 data index 替换为真实的 data index
        # indices_buckets[bucket_id].samples = [normal_indexes[indice] for indice in indices_buckets[bucket_id].samples]

        # 将桶分配给选定的DP rank
        # fwd_flops_for_dp_per_m[min_flops_dp_rank].append(used_fwd_flops[bucket_id])
        # bwd_flops_for_dp_per_m[min_flops_dp_rank].append(used_bwd_flops[bucket_id])
        fwd_flops_for_dp_per_m[min_flops_dp_rank].append(bucket_tmp.fwd_flops)
        bwd_flops_for_dp_per_m[min_flops_dp_rank].append(bucket_tmp.bwd_flops)
        # correction for memory simulator
        seq_len_for_dp_per_m[min_flops_dp_rank].append([bucket_tmp.seq_len_sum // config.min_hybrid_context_parallel_size * config.context_parallel_size])
        buckets_for_dp[min_flops_dp_rank].append(bucket_tmp)
        sample_ids_for_dp[min_flops_dp_rank].append(bucket_tmp.samples)

        # 更新DP rank的负载统计
        fwd_flops_sum_per_dp_this_m[min_flops_dp_rank] += (bucket_tmp.fwd_flops[0])
        bucket_used_num_per_dp_this_m[min_flops_dp_rank] += 1
        # for ttt in bucket_used_num_per_dp_this_m:
        #     print_rank0(ttt, end='\t')
        # print_rank0("")

    # print_rank0(f"aft asign normal bucket, {dp_size_for_current_m=}, {bucket_used_num_per_dp_this_m=}")
    
    for dp_rank in range(len(buckets_for_dp)):
        # print_rank0(f"rank {torch.distributed.get_rank()} bucket num for dp{len(buckets_for_dp[dp_rank])}")
        # num_fused = sum(1 for b in buckets_for_dp[dp_rank] if not isinstance(b, SplitBucket))
        for bucket_i, bucket in enumerate(buckets_for_dp[dp_rank]):
            bucket.num_split_bucket_this_dp = num_split_for_dp[dp_rank]
            # if isinstance(bucket, SplitBucket):
            #     # print_rank0(f"{dp_rank=}, {bucket_i=}, {bucket.fwd_flops=}")
            # else:
            #     # print_rank0(f"{dp_rank=}, {bucket_i=}, {bucket.samples_fwd_flops=} {len(bucket.samples)=}")
    assert len(buckets_for_dp) == len(sample_ids_for_dp), f"{len(sample_ids_for_dp)=}, {len(buckets_for_dp)=}"
    return fwd_flops_for_dp_per_m, bwd_flops_for_dp_per_m, buckets_for_dp, sample_ids_for_dp, seq_len_for_dp_per_m, empty_bucket_flag



def fwd_flops_update_rule(bucket, index, all_density, all_lengths, all_flops):
    if bucket.seq_len_sum + all_lengths[index] > bucket.target_length:    # add memory limit.
        return None

    new_fwd_flops = bucket.fwd_flops + all_flops[1][index]
    return new_fwd_flops * (new_fwd_flops / bucket.target_flops)


def length_update_rule(bucket, index, all_density, all_lengths, all_flops):
    return ((bucket.seq_len_sum + all_lengths[index]) - bucket.target_length)** 2 / bucket.target_length**2


class UpdateRule(enum.Enum):
    DENSITY = 1
    FW_FLOPS = 2
    LENGTH = 3


update_rule_mapping = {
    UpdateRule.FW_FLOPS: fwd_flops_update_rule,
    UpdateRule.LENGTH: length_update_rule
}


def fwd_flops_to_bwd_flops(pre_attn_fwd_time, attn_fwd_time, post_attn_fwd_time, mlp_fwd_time):
    attn_bwd_time = 2.77 * attn_fwd_time
    pre_attn_bwd_time = 2.7 * pre_attn_fwd_time
    post_attn_bwd_time = 2.7 * post_attn_fwd_time
    mlp_bwd_time = 2.7 * mlp_fwd_time

    return pre_attn_bwd_time, attn_bwd_time, post_attn_bwd_time, mlp_bwd_time


def attention_tflops(s, h, scale):
    # NOTE: only consider forward tflops
    s2 = s**2
    tflops = 2 * 2 * s2 * h / 1e12 * scale
    return tflops


def linear_tflops(s, h1, h2):
    # NOTE: only consider forward tflops
    tflops = 2 * s * h1 * h2 / 1e12
    return tflops


def TFLOPs(s1, config):
    """
        Only calculate one block TFLOPs here.
    """
    scale = 0.5

    ####### forward tflops ########
    gemm_fwd_tflops = linear_tflops(s1, config.hidden_size, config.hidden_size)
    attn_fwd_tflops = attention_tflops(s1, config.hidden_size, scale)

    pre_attn_fwd_tflops = 3 * gemm_fwd_tflops
    if config.num_query_groups is not None:
        pre_attn_fwd_tflops = gemm_fwd_tflops + 2 * gemm_fwd_tflops / config.num_query_groups
    post_attn_fwd_tflops = gemm_fwd_tflops
    mlp_fc1_h = config.ffn_hidden_size * 2 if config.gated_linear_unit else config.ffn_hidden_size
    mlp_fc2_h = config.ffn_hidden_size
    mlp_fc1_fwd_tflops = linear_tflops(s1, config.hidden_size, mlp_fc1_h)
    mlp_fc2_fwd_tflops = linear_tflops(s1, mlp_fc2_h, config.hidden_size)

    fwd_tflops = pre_attn_fwd_tflops + attn_fwd_tflops + post_attn_fwd_tflops + mlp_fc1_fwd_tflops + mlp_fc2_fwd_tflops
    
    ####### backward tflops ########
    pre_attn_bwd_tflops, attn_bwd_tflops, post_attn_bwd_tflops, mlp_bwd_tflops = \
        fwd_flops_to_bwd_flops(pre_attn_fwd_tflops, attn_fwd_tflops, post_attn_fwd_tflops, mlp_fc1_fwd_tflops + mlp_fc2_fwd_tflops)

    bwd_tflops = pre_attn_bwd_tflops + attn_bwd_tflops + post_attn_bwd_tflops + mlp_bwd_tflops

    ####### recompute tflops ########
    if config.recompute_granularity == "full":
        bwd_tflops += fwd_tflops
    else:
        # TODO: add other recompute method here.
        pass

    tot_tflops = fwd_tflops + bwd_tflops
    return tot_tflops, fwd_tflops, bwd_tflops


def compute_ratios(combination, PP):
    # PP = mpu.get_pipeline_model_parallel_world_size()
    VPP = 1
    ratios = []
    for num_m in range(1, len(combination)+1):
        ratio = num_m * PP * (PP * VPP + PP - 1) / (num_m * PP * VPP + PP - 1)
        ratios.append(ratio)
    return ratios


class Bucket:
    def __init__(self, target_flops, target_density, target_length, bucket_id, cp_size, samples, fwd_flops=0, bwd_flops=0, seq_len_sum=0, dp_index=-1):
        self.bucket_id = bucket_id
        self.samples = samples
        self.cp_size = cp_size
        self.target_flops = target_flops
        self.target_density = target_density
        self.target_length = target_length
        self.current_density = 0
        self.fwd_flops = fwd_flops
        self.bwd_flops = bwd_flops
        self.seq_len_sum = seq_len_sum
        self.dp_index = dp_index
        self.type = "Bucket"

    def __str__(self):
        return (
            f"Bucket {self.bucket_id}:\n"
            f"  Target Flop: {self.target_flops}\n"
            f"  Target Density: {self.target_density}\n"
            f"  Target Length: {self.target_length}\n"
            f"  Current Density: {self.current_density}\n"
            f"  Forward Flops: {self.fwd_flops}\n"
            f"  Backward Flops: {self.bwd_flops}\n"
            f"  Sequence Length Sum: {self.seq_len_sum}\n"
            f"  Samples: {self.samples}\n"
        )


def create_buckets(num_buckets, avg_fwd_flops_with_m, max_seq_len_for_fuse):
    total_bucket_num = 0
    all_buckets = []

    for i in range(len(num_buckets)):
        for j in range(num_buckets[i]):
            target_density =  avg_fwd_flops_with_m[i] / max_seq_len_for_fuse
            all_buckets.append(
                Bucket(
                    target_flops = avg_fwd_flops_with_m[i], 
                    target_density = target_density, 
                    target_length = max_seq_len_for_fuse, 
                    bucket_id = total_bucket_num,
                    cp_size = 1,
                    samples = [],
                )
            )
            total_bucket_num += 1
    
    return total_bucket_num, all_buckets


def assign_samples_to_buckets(
    sorted_indices,
    buckets,
    all_density,
    all_lengths,
    all_flops,
    update_rule=None,
    remaining_sample_indices=None,
    print_score=False,
):

    preassigned_samples = []
    if update_rule is UpdateRule.DENSITY:
        raise Exception()
        print_rank0("using density update rule")
        pre_assign_sample_to_empty_bucket(sorted_indices, buckets, all_density, all_lengths, all_flops, remaining_sample_indices, preassigned_samples)
    else:
        assert len(preassigned_samples) == 0
    update_rule = update_rule_mapping[update_rule]

    for index in sorted_indices:
        if index in preassigned_samples:
            continue

        min_score = float('inf')
        target_bucket = None

        score = None
        for bucket in buckets:
            score = update_rule(bucket, index, all_density, all_lengths, all_flops)
            if score is not None and score < min_score:
                min_score = score
                target_bucket = bucket

        if target_bucket is not None:
            target_bucket.fwd_flops += all_flops[1][index]
            target_bucket.bwd_flops += (2 * all_flops[1][index])    # TODO(wuguohao): use more precisely bwd_flops
            target_bucket.seq_len_sum += all_lengths[index]
            target_bucket.samples.append(index)
            remaining_sample_indices.remove(index)
            # if torch.distributed.get_rank() == 0: print(f"pop {index=}, {len(remaining_sample_indices)=}, length_rule={update_rule == length_update_rule}, flops_rule={update_rule == fwd_flops_update_rule} {remaining_sample_indices=}")
        else:
            if update_rule == length_update_rule:
                if torch.distributed.get_rank() == 0: print(f"skip {index=}, {score=}, {min_score=} {len(buckets)=}, {len(remaining_sample_indices)=}, length_rule={update_rule == length_update_rule}, flops_rule={update_rule == fwd_flops_update_rule} ")

    return remaining_sample_indices


def nearest_pow2(n: int) -> int:
    """
    将正整数 n 四舍五入到最接近的 2 的幂。
    n < 1 时返回 1。
    """
    if n < 1:
        return 1
    # lower = 2^(⌊log2 n⌋)
    lower = 1 << (n.bit_length() - 1)
    # upper = 2^(⌈log2 n⌉)
    upper = 1 << n.bit_length()
    # 距离较小者
    return lower if (n - lower) < (upper - n) else upper


def fill_bucket_with_samples(
    curr_except_index,
    sorted_indices,
    target_flops,
    all_flops,
    all_lengths,
    max_seq_len,
    remaining_sample_indices=None,
    total_num=0,
    consumed_num_buckets=0,
    assign_all_sample_to_except_bucket_flag=False,
):
    # assume sorted_indices is sorted by fwd flops in reversed order.
    selected_indices = []
    selected_fwd_flops = []
    selected_bwd_flops = []
    selected_lengths = []
    remained_flops = target_flops
    # print_rank0(f"###{max_num_samples_to_fill=}")
    length_sum = all_lengths[curr_except_index]
    for index in sorted_indices:
        # if max_num_samples_to_fill == 0: break
        sample_fwd_flops = all_flops[1][index]
        # sample_bwd_flops = all_flops[2][index]
        sample_bwd_flops = all_flops[2][index]  # TODO(wuguohao): more precisely bwd_flops
        extra_limit = total_num < len(remaining_sample_indices) and length_sum < (max_seq_len * consumed_num_buckets)
        extra_limit = (assign_all_sample_to_except_bucket_flag) or extra_limit  # skip extra_limit if `assign_all_sample_to_except_bucket_flag` is True

        exceed_ratio = 1.05
        # if assign_all_sample_to_except_bucket_flag:
        #     exceed_ratio = 1.5
        if sample_fwd_flops < remained_flops * exceed_ratio and extra_limit: # TODO: consume num buckets * max seq len
            # if torch.distributed.get_rank() == 0: print(f"{target_flops=}, {index=}, {sample_fwd_flops=}, {sample_bwd_flops=}")
            remained_flops -= sample_fwd_flops
            selected_indices.append(index)
            selected_fwd_flops.append(sample_fwd_flops)
            selected_bwd_flops.append(sample_bwd_flops)
            selected_lengths.append(all_lengths[index])
            remaining_sample_indices.remove(index)
            length_sum += all_lengths[index]
            # max_num_samples_to_fill -= 1

    selected_flops = [selected_fwd_flops, selected_bwd_flops]
    
    return selected_indices, selected_flops, selected_lengths, remained_flops, remaining_sample_indices


class PipelineAwareBalancedHybridCPscheduler(BaseScheduler):
    
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_len_per_rank = config.max_seqlen_per_dp_cp_rank
        self.num_subsamples = 0
        self.num_subsamples_processed = 0
        self.free_resources = []
        self.total_hdp_gpus = parallel_state.get_data_parallel_world_size(
            with_context_parallel=True
        )

    @lru_cache(maxsize=128)
    def get_total_workload(self, seq_length: int, cp_size: Optional[int] = None):
        """
        seq_length: sequence length of a sub-sample
        cp_size: total number of CP ranks working on this sub-sample

        Note:
        This function is used to estimate the relative workload intensity
        of a sub-sample. This is not meant to be an accurate flops calculator.

        Returns: workload of a sub-sample
        """
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size

    def get_groups_and_subsamples(self, sample_id_seqlens, config, return_cp_sizes=False):
        """
        This function recursively forms groups of sub-samples such that all DPxCP ranks
        have a roughly balanced workload in the group.
        """
        groups = []
        sample_id_groups = []
        cp_sizes = []
        # We assign a sample_id to each sub-sample in order to track assignment to each GPU.
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        # while sample_id_seqlens:
        #     mb, sample_id_seqlens, exec_times, sample_ids = self.next_hdp_group(
        #         sample_id_seqlens, self.get_total_workload, self.total_hdp_gpus, config=config
        #     )
        #     groups.append(mb)
        #     if len(sample_ids) < self.total_hdp_gpus:
        #         sample_ids.extend([] * (self.total_hdp_gpus - len(sample_ids)))
        #     sample_id_groups.append(sample_ids)

        _, _, best_indices_buckets, best_sample_ids, best_dp_combination, _ = self.next_hdp_group(
            sample_id_seqlens, self.get_total_workload, self.total_hdp_gpus, config=config
        )

        # print(best_indices_buckets[-1][0][0])
        # breakpoint()

        mi = -1
        for i in range(len(best_indices_buckets)):
            if len(best_indices_buckets[i]) > 0:
                mi = i
                break
        assert mi != -1
        best_sample_ids = best_sample_ids[mi]
        best_indices_buckets = best_indices_buckets[mi]

        # print(f"{len(best_indices_buckets)=}, {len(best_sample_ids)=}")
        assert len(best_indices_buckets) == len(best_sample_ids)
        # print(f"{best_sample_ids=}, {len(best_indices_buckets)=}, {len(best_indices_buckets[0])=}, {len(best_indices_buckets[1])=}")
        # breakpoint()

        def transpose_2d_list(matrix):
            return [list(row) for row in zip(*matrix)]

        local_sample_id_groups = transpose_2d_list(best_sample_ids)
        local_best_indices_buckets = transpose_2d_list(best_indices_buckets)
        # groups = 
        min_hybrid_context_parallel_size = config.min_hybrid_context_parallel_size
        for microbatch_idx in range(len(local_sample_id_groups)):
            sample_id_groups.append([])
            groups.append([])
            cp_sizes.append([])
            dpxcp = len(local_sample_id_groups[microbatch_idx]) * min_hybrid_context_parallel_size
            for dp_rank in range(dpxcp):
                # for min_hybrid_context_parallel_rank in range(min_hybrid_context_parallel_size):
                sample_id_groups[microbatch_idx].append([])
                groups[microbatch_idx].append([])
                cp_sizes[microbatch_idx].append([])
                origin_dp_rank = dp_rank // min_hybrid_context_parallel_size
                # if torch.distributed.get_rank() == 0: print(f"{microbatch_idx=}, {dp_rank=}, {origin_dp_rank=}, {local_sample_id_groups[microbatch_idx][origin_dp_rank]=}")
                for local_sample_idx in local_sample_id_groups[microbatch_idx][origin_dp_rank]:
                    sample_id_groups[microbatch_idx][dp_rank].append(sample_id_seqlens[local_sample_idx][0])
                    groups[microbatch_idx][dp_rank].append(sample_id_seqlens[local_sample_idx][1])
                    final_cp_size = local_best_indices_buckets[microbatch_idx][origin_dp_rank].cp_size * min_hybrid_context_parallel_size
                    cp_sizes[microbatch_idx][dp_rank].append(final_cp_size)

        # if torch.distributed.get_rank() == 0: print(f"{sample_id_groups=}")
        # if torch.distributed.get_rank() == 0: print(f"{cp_sizes=}")
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        # 示例
        # nested_list = [1, [2, 3], [4, [5, 6]], 7]
        # print(flatten(nested_list))  # [1, 2, 3, 4, 5, 6, 7]

        # breakpoint()

        if return_cp_sizes:
            return groups, sample_id_groups, cp_sizes

        return groups, sample_id_groups
    def split_sample(
        self,
        num_buckets: List[int],
        avg_fwd_flops_with_m: List[float],
        all_lengths,
        all_flops,
        except_indexes,
        normal_indexes,
        combination,
        DP, PP, UP, TP,
        max_split_size,
        max_seq_len,
        config,
    ):
        num_layers = config.num_layers               # 模型层数
        hidden_size = config.hidden_size             # 隐藏层大小
        num_heads = config.num_attention_heads       # 注意力头数
        assert hidden_size % num_heads == 0, "hidden_size should be divisible by num_heads"
        head_dim = hidden_size // num_heads        # 每个注意力头的维度
        ffn_size = config.ffn_hidden_size           # FFN层隐藏大小

        # 初始化特殊序列的桶分配结构
        except_buckets = [[] for _ in range(len(num_buckets))]  # 每个m值对应的特殊序列桶
        except_bucket_num = 0                      # 特殊序列桶计数器
        except_bucket_m_per_sample = []            # 记录每个样本分配到的m值
        except_bucket_dp_per_sample = []           # 记录每个样本分配到的dp值
        except_bucket_num_per_sample = []          # 记录每个样本分割的桶数量

        # 计算每个 m 下单个 dp 的桶数(相同 m 的不同 dp 的桶数相等)
        bucket_num_per_dp_per_m = []
        # import pdb;pdb.set_trace()
        for i in range(len(num_buckets)):
            if combination[i] > 0:
                assert num_buckets[i] % combination[i] == 0, f"{i=}, {num_buckets[i]=}, {combination[i]=}"
                bucket_num_per_dp_per_m.append(num_buckets[i] // combination[i])
            else:
                bucket_num_per_dp_per_m.append(0)
        # print_rank0(f"{bucket_num_per_dp_per_m=}")

        # 维护不同 dp 当前剩余桶数，使用该桶数去做大 UP
        remain_buckets_num_per_dp_per_m = []
        for i in range(len(num_buckets)):
            if combination[i] > 0:
                assert num_buckets[i] % combination[i] == 0, f"{i=}, {num_buckets[i]=}, {combination[i]=}"
                # import pdb;pdb.set_trace()
                remain_buckets_num_per_dp_per_m.append([num_buckets[i] // combination[i]] * combination[i])
            else:
                remain_buckets_num_per_dp_per_m.append([])

        # 遍历所有需要独占一路 DP 的序列
        single_sample_indexes = []   # 去掉需要独占一个 DP 的样本后的 except_indexes
        combination_used = [0] * len(combination)

        # 重新计算 桶的容积
        sum_fwd_flops = sum([all_flops[1][idx] for idx in except_indexes if idx not in single_sample_indexes]) + \
            sum([all_flops[1][idx] for idx in normal_indexes])
            
        ratios = compute_ratios(combination, PP=PP)
        avg_fwd_flops_with_m_new = []
        total_num = sum([(combination[j]-combination_used[j]) * ratios[j] for j in range(len(combination))])  # TODO: total num need to - exceed_buckets num
        mean_fwd_flops_with_m = sum_fwd_flops / total_num
        for i in range(1, len(combination)+1):
            avg_fwd_flops_with_m_new.append(mean_fwd_flops_with_m * ratios[i - 1] / i / PP)

        avg_fwd_flops_with_m = avg_fwd_flops_with_m_new

        non_zero_combination = [(combination[idx]-combination_used[idx]) != 0 for idx in range(len(combination))]
        first_non_zero_m = 1 + non_zero_combination.index(True)
        threshold = 2 * sum_fwd_flops / (first_non_zero_m * (DP-len(single_sample_indexes)) * PP)

        consumed_num_buckets_backup = {}
        consumed_num_buckets_raw_backup = {}
        for idx, index in enumerate(except_indexes):
            find_bucket = False
            for i in range(len(num_buckets)):
                # 只考虑有剩余桶的m值
                if combination[i] > 0:
                    # 计算当前序列需要的桶数量(向上取整)
                    consumed_num_buckets_raw = math.ceil(all_flops[1][index] / avg_fwd_flops_with_m[i])
                    remain_num_split_sample = len(except_indexes) - 1 - idx
                    consumed_num_buckets = min(nearest_pow2(consumed_num_buckets_raw), max_split_size, DP//config.min_hybrid_context_parallel_size, num_buckets[i]-remain_num_split_sample)
                    consumed_num_buckets_raw_backup[index] = consumed_num_buckets_raw
                    consumed_num_buckets_backup[index] = consumed_num_buckets
                    # 更新剩余桶数量
                    num_buckets[i] -= consumed_num_buckets
                    find_bucket = True
                    break


        assign_all_sample_to_except_bucket_flag = False
        assert sum(num_buckets) >= 0
        if sum(num_buckets) == 0:
            assign_all_sample_to_except_bucket_flag = True
        # if torch.distributed.get_rank() == 0: print(f"{num_buckets=}\n{consumed_num_buckets_raw_backup.keys()=}\n{consumed_num_buckets_raw_backup.values()=}\n{consumed_num_buckets_backup.keys()=}\n{consumed_num_buckets_backup.values()=}\n{except_indexes=}")
        
        for index in except_indexes:
            find_bucket = False
            for i in range(len(num_buckets)):
                # 只考虑有剩余桶的m值
                if combination[i] > 0:
                    # 计算当前序列需要的桶数量(向上取整)
                    # consumed_num_buckets_raw = math.ceil(all_flops[1][index] / avg_fwd_flops_with_m[i])
                    # consumed_num_buckets = min(min(nearest_pow2(consumed_num_buckets_raw), max_split_size), DP)
                    consumed_num_buckets = consumed_num_buckets_backup[index]
                    # print(f"{index=}, {i=}, {consumed_num_buckets_raw=}, {consumed_num_buckets=}")
                    remained_flops = consumed_num_buckets * avg_fwd_flops_with_m[i] - all_flops[1][index]

                    # choose the CP interval
                    max_value = -1
                    max_left = max_right = -1
                    max_indexes = [-1] * consumed_num_buckets
                    for j in range(combination[i]):
                        left = (j // consumed_num_buckets) * consumed_num_buckets
                        right = (j // consumed_num_buckets + 1) * consumed_num_buckets
                        min_value_this_interval = 10000000
                        # for dp size not divisible by consumed_num_buckets, continue to skip this search space
                        if right > len(remain_buckets_num_per_dp_per_m[i]):
                            continue

                        for k in range(left, right): #left close right close
                            min_value_this_interval = min(min_value_this_interval, remain_buckets_num_per_dp_per_m[i][k])
                        if max_value < min_value_this_interval:
                            max_value = min_value_this_interval
                            max_left = left
                            max_right = right
                            max_indexes = list(range(left, right))
                    
                    normal_indexes_copy = copy.deepcopy(normal_indexes)
                    selected_indices, selected_flops, selected_lengths, remained_flops, remaining_sample_indices = fill_bucket_with_samples(index, normal_indexes, remained_flops, all_flops, all_lengths, max_seq_len, normal_indexes_copy, total_num, consumed_num_buckets, assign_all_sample_to_except_bucket_flag)
                    # print(f"\n{len(selected_indices)=}, {selected_indices=}\n{len(remaining_sample_indices)=}, {remaining_sample_indices=}\n{sum(selected_lengths)=}, {sum(selected_flops[0])=}, {remained_flops=}, {all_lengths[index]=}, {max_seq_len*consumed_num_buckets=}")
                    normal_indexes = remaining_sample_indices
                    for j in range(consumed_num_buckets):
                        remain_buckets_num_per_dp_per_m[i][max_indexes[j]] -= 1

                    assert len(max_indexes) == consumed_num_buckets, f"{len(max_indexes)=}, {consumed_num_buckets=}"
                    # 将分割后的序列片段分配到各个桶中
                    for j in range(consumed_num_buckets):
                        bucket_fwd_flops = all_flops[1][index] + sum(selected_flops[0])
                        bucket_bwd_flops = (3 * all_flops[1][index]) + sum(selected_flops[1])   # TODO(wuguohao): more precisely bwd_flops
                        bucket_length = all_lengths[index] + sum(selected_lengths)
                        bucket_tmp = [index] + selected_indices
                        #shenglong target_flops=1 to handle except use all buckets
                        except_buckets[i].append(
                            Bucket(
                                bucket_id=except_bucket_num,
                                samples=bucket_tmp,
                                cp_size=consumed_num_buckets,
                                fwd_flops=bucket_fwd_flops/consumed_num_buckets,
                                bwd_flops=bucket_bwd_flops/consumed_num_buckets,
                                seq_len_sum=bucket_length,
                                target_flops=1, target_density=0, target_length=0,
                                dp_index=max_indexes[j],
                            )
                        )
                        except_bucket_num += 1  # 递增桶计数器

                    # 更新剩余桶数量
                    # num_buckets[i] -= consumed_num_buckets
                    # 记录分配信息
                    except_bucket_num_per_sample.append(consumed_num_buckets)
                    except_bucket_m_per_sample.append(i)
                    except_bucket_dp_per_sample.append(max_indexes)

                    find_bucket = True
                    break  # 成功分配到桶中，跳出循环

            if not find_bucket:
                raise NotImplementedError("not found a bucket for the sample")
        
        assert len(except_bucket_m_per_sample) == len(except_bucket_num_per_sample), f"{len(except_bucket_m_per_sample)=}, {len(except_bucket_num_per_sample)=}"
        return except_buckets, num_buckets, except_bucket_num_per_sample, except_bucket_m_per_sample, except_bucket_dp_per_sample, except_indexes, normal_indexes, avg_fwd_flops_with_m

    def next_hdp_group(
        self,
        sample_seqlens: List[Tuple[int, int]],  # List of (sample_id, sequence_length) tuples
        compute_estimator: Callable[[int], float],
        total_gpus: int,
        delta: float = 0.05,  # balance slack (e.g. 5 %)
        strategy: str = "dp",  # "dp" or "pp"
        eps_bucket: float = 0.10,  # ε target for bucket balance
        config = None,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:

        DP = parallel_state.get_data_parallel_world_size()
        PP = parallel_state.get_pipeline_model_parallel_world_size()
        UP = parallel_state.get_context_parallel_world_size()
        TP = parallel_state.get_tensor_model_parallel_world_size()

        VPP = 1
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            VPP = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        # if torch.distributed.get_rank() == 0:
        #     breakpoint()
        # torch.distributed.barrier()

        max_split_size = config.max_hybrid_context_parallel_size // config.min_hybrid_context_parallel_size
        max_seq_len = config.max_seqlen_per_dp_cp_rank

        all_lengths = [sample_seqlens[i][1] for i in range(len(sample_seqlens))]
        
        all_flops = []
        all_tot_flops = []
        all_fwd_flops = []
        all_bwd_flops = []
        for idx in range(len(all_lengths)):
            length = all_lengths[idx]
            flops = TFLOPs(length, config)
            all_tot_flops.append(flops[0])
            all_fwd_flops.append(flops[1])
            all_bwd_flops.append(flops[2])
        all_flops.append(all_tot_flops)
        all_flops.append(all_fwd_flops)
        all_flops.append(all_bwd_flops)

        all_density = [all_flops[1][i] / all_lengths[i] for i in range(len(all_lengths))]
        best_max_seq_per_m = 0
        sum_fwd_flops = sum(all_flops[1])

        assert len(all_lengths) == len(all_flops[1])

        def dynamic_loops_product(limits):
            min_max_flops_sum_per_iter = sys.float_info.max / 10.0
            best_indices_buckets = []
            best_sample_ids = []
            best_dp_combination = []

            assert DP % config.min_hybrid_context_parallel_size == 0
            limit_item = DP // config.min_hybrid_context_parallel_size

            for idx, limit in enumerate(limits):
                combination = [0] * len(limits)
                combination[idx] = limit
                if sum(combination) != limit_item:
                    continue

                num_buckets = [PP * i * combination[i - 1] for i in range(1, len(combination)+1)]
                num_buckets_sum = sum(num_buckets)

                if num_buckets_sum > len(all_lengths):
                    print(f"continue due to num_buckets_sum {num_buckets_sum=} > len(all_lengths) {len(all_lengths)=}")
                    continue

                ratios = compute_ratios(combination, PP=PP)
                avg_fwd_flops_with_m = []
                total_num = sum(combination[j] * ratios[j] for j in range(len(combination)))
                mean_fwd_flops_with_m = sum_fwd_flops / total_num

                for i in range(1, len(combination)+1):
                    avg_fwd_flops_with_m.append(mean_fwd_flops_with_m * ratios[i - 1] / i / PP)

                st_time = time.time()
                indices_buckets, sample_ids, max_flops_sum_per_iter, max_seq_per_m, used_flops = \
                    self.solver(all_lengths, all_flops, all_density, num_buckets, avg_fwd_flops_with_m, combination, DP, PP, UP, TP, VPP, max_seq_len, max_split_size, config)
                ed_time = time.time()
                if torch.distributed.get_rank() == 0: print(f"solver cost time :{ed_time-st_time}s")
                
                if max_flops_sum_per_iter < min_max_flops_sum_per_iter:
                    min_max_flops_sum_per_iter = max_flops_sum_per_iter
                    best_indices_buckets = indices_buckets
                    best_sample_ids = sample_ids
                    best_dp_combination = combination
                    # if torch.distributed.get_rank() == 0:
                    #     print(f"{best_dp_combination=}\n{best_indices_buckets=}\n{best_sample_ids=}")

            return min_max_flops_sum_per_iter, best_indices_buckets, best_sample_ids, best_dp_combination

        search_space = config.search_space
        assert DP % config.min_hybrid_context_parallel_size == 0
        limit_item = DP // config.min_hybrid_context_parallel_size
        # limits = [limit_item] * search_space
        if isinstance(search_space, int):
            limits = [limit_item] * search_space
        elif isinstance(search_space, list):
            limits = [0] * max(search_space)
            for idx in search_space:
                limits[idx] = limit_item
        else:
            raise Exception(f"`search_space` should be int or list, but {type(search_space)} found.")

        min_max_flops_sum_per_iter, best_indices_buckets, best_sample_ids, best_dp_combination = dynamic_loops_product(limits)
        if torch.distributed.get_rank() == 0: print(f"{best_dp_combination=}")
        # assert all DP have the same num_microbatch
        sum_best_dp_combination = sum(best_dp_combination)
        best_m = -1
        for idx, num_dp in enumerate(best_dp_combination):
            if num_dp == sum_best_dp_combination:
                best_m = idx
                break
        assert best_m != -1, f"{best_dp_combination=}"

        if not best_dp_combination:
            raise Exception()

        best_var_m = 0
        return min_max_flops_sum_per_iter, best_max_seq_per_m, best_indices_buckets, best_sample_ids, best_dp_combination, best_var_m

    def solver(
        self,
        all_lengths: List[int],
        all_flops: List[List[float]],
        all_density: List[float],
        num_buckets: List[int],
        avg_fwd_flops_with_m: List[float],
        combination,
        DP, PP, UP, TP, VPP,
        max_seq_len,
        max_split_size,
        config,
    ):
        except_indexes = []
        normal_indexes = []

        non_zero_combination = [combination[idx] != 0 for idx in range(len(combination))]
        first_non_zero_m = 1 + non_zero_combination.index(True)

        sum_fwd_flops = sum(all_flops[1])
        threshold = 1.3 * sum_fwd_flops / (first_non_zero_m * DP * PP)
        for idx in range(len(all_flops[1])):
            if  all_flops[1][idx] > threshold:
                except_indexes.append(idx)
            else:
                normal_indexes.append(idx)
        # if torch.distributed.get_rank() == 0:
        #     print(f"\n{except_indexes=}")
        #     except_flops = []
        #     for idx in except_indexes:
        #         except_flops.append(all_flops[1][idx])
        #     print(f"{except_indexes=}\n{except_flops=}")
        except_indexes = sorted(except_indexes, key=lambda x: all_flops[1][x], reverse=True)
        normal_indexes = sorted(normal_indexes, key=lambda x: all_flops[1][x], reverse=True)

        except_buckets, num_buckets, except_bucket_num_per_sample, except_bucket_m_per_sample, except_bucket_dp_per_sample, except_indexes, normal_indexes, avg_fwd_flops_with_m = \
            self.split_sample(num_buckets, avg_fwd_flops_with_m, all_lengths, all_flops, except_indexes, normal_indexes, combination, DP, PP, UP, TP, max_split_size, max_seq_len, config)

        sum_remained_flops = sum([all_flops[1][index] for index in normal_indexes])

        # for the case that except indexes take all buckets
        if sum(num_buckets) != 0:
            max_seq_len_for_fuse = sum([all_lengths[idx] for idx in normal_indexes]) / sum(num_buckets)
        else:
            max_seq_len_for_fuse = 0
        

        if max_seq_len_for_fuse == 0:
            assert len(normal_indexes) == 0
        total_bucket_num, all_buckets = create_buckets(num_buckets, avg_fwd_flops_with_m, max_seq_len_for_fuse)
        sorted_indices_fwdflops = sorted(normal_indexes, key=lambda x: all_flops[1][x], reverse=True)
        sorted_all_buckets_fwd_flops = sorted(all_buckets, key=lambda bucket: bucket.fwd_flops)
        all_sample_index_copy = copy.deepcopy(sorted_indices_fwdflops)

        all_sample_index_copy_bef_flops = copy.deepcopy(all_sample_index_copy)
        all_sample_index_copy = assign_samples_to_buckets(sorted_indices_fwdflops,
                                    sorted_all_buckets_fwd_flops,
                                    all_density,
                                    all_lengths,
                                    all_flops,
                                    update_rule=UpdateRule.FW_FLOPS,
                                    remaining_sample_indices=all_sample_index_copy)
    
        # If there are some leftover of the samples 
        # (e.g. if put the sample in any of the bucket will cause the bucket exceed the memory limit),
        # we will use the length update rule to assign those samples to the bucket.
        # The all_sample_index_copy should contain only a few samples. Sorting might be unnecessary.
        sorted_indices_length = sorted(all_sample_index_copy, key=lambda x: all_lengths[x], reverse=True)
        sorted_all_buckets_length = sorted(all_buckets, key=lambda bucket: bucket.seq_len_sum)

        if len(all_sample_index_copy) > 0:
            all_sample_index_copy_bef_len = copy.deepcopy(all_sample_index_copy)
            all_sample_index_copy = assign_samples_to_buckets(sorted_indices_length, sorted_all_buckets_length, all_density, all_lengths, all_flops, update_rule=UpdateRule.LENGTH, remaining_sample_indices=all_sample_index_copy, print_score=True)

        assert len(all_sample_index_copy) == 0, f"sample {all_sample_index_copy} is not assigned to any bucket."

        indices_buckets = [[] for _ in range(total_bucket_num)]
        used_flops = [0.0] * total_bucket_num
        used_fwd_flops = [0.0] * total_bucket_num
        used_bwd_flops = [0.0] * total_bucket_num
        max_seq_per_m = 0
        seq_per_m = []
        
        for bucket in sorted_all_buckets_fwd_flops:
            bucket_id = bucket.bucket_id 
            indices_buckets[bucket_id] = bucket
            used_flops[bucket_id] = bucket.fwd_flops + bucket.bwd_flops
            used_fwd_flops[bucket_id] = bucket.fwd_flops
            used_bwd_flops[bucket_id] = bucket.bwd_flops
            max_seq_per_m = max(bucket.seq_len_sum, max_seq_per_m)
            seq_per_m.append(bucket.seq_len_sum)

        indices_buckets_2d = [[] for _ in range(len(num_buckets))]
        sample_ids_2d = [[] for _ in range(len(num_buckets))]
        new_cnt = 0
        max_sum_per_iter = 0.0
        rets = [0.0] * DP
        thread_cnt = 0

        max_iter_sum_among_dp_list = []
        for i in range(len(num_buckets)):
            if len(except_buckets[i]) + num_buckets[i] == 0:
                assert combination[i] == 0, f"{combination=}, {num_buckets=}, {len(except_buckets[i])=}"
                continue

            total_buckets_for_current_m = num_buckets[i] + len(except_buckets[i])
            num_m = i + 1
            bucket_num_per_dp_curr_m = num_m * PP
            assert total_buckets_for_current_m % bucket_num_per_dp_curr_m == 0, f"{total_buckets_for_current_m=}, {bucket_num_per_dp_curr_m=}"
            dp_size_for_current_m = total_buckets_for_current_m // bucket_num_per_dp_curr_m

            buckets_for_current_m = []
            for j in range(num_buckets[i]):
                buckets_for_current_m.append([used_flops[new_cnt], new_cnt, used_fwd_flops[new_cnt]])
                new_cnt += 1

            buckets_for_current_m.sort(key=lambda x: x[2])

            fwd_flops_for_dp_per_m, bwd_flops_for_dp_per_m, buckets_for_dp, sample_ids_for_dp, seq_len_for_dp_per_m, empty_bucket_flag = greedy_assign_bucket_to_dp(i, indices_buckets, normal_indexes, except_buckets, except_bucket_num_per_sample, except_bucket_m_per_sample, except_bucket_dp_per_sample, buckets_for_current_m, dp_size_for_current_m, used_flops, used_fwd_flops, used_bwd_flops, bucket_num_per_dp_curr_m, all_flops, all_lengths, combination, config)

            for j in range(len(buckets_for_dp)):
                indices_buckets_2d[i].append(buckets_for_dp[j])
                sample_ids_2d[i].append(sample_ids_for_dp[j])
            
            assert len(indices_buckets_2d) == len(sample_ids_2d), f"{len(indices_buckets_2d)=}, {len(sample_ids_2d)=}"

            bubble_time_list = []
            if empty_bucket_flag:
                print(f"error, found empty bucket, skip")
                max_sum_per_iter = sys.float_info.max / 10.0
            else:
                for m in range(len(fwd_flops_for_dp_per_m)):
                    total_bucket_num_for_current_dp = len(fwd_flops_for_dp_per_m[m])
                    forward_cost = [fwd_flops_for_dp_per_m[m][k][0] for k in range(len(fwd_flops_for_dp_per_m[m]))]
                    backward_cost = [bwd_flops_for_dp_per_m[m][k][0] for k in range(len(fwd_flops_for_dp_per_m[m]))]
                    seq_len_for_dp = seq_len_for_dp_per_m[m]
                    communication_cost = [0.0] * len(fwd_flops_for_dp_per_m[m])

                    forward_cost_cmp = []
                    backward_cost_cmp = []
                    assert len(fwd_flops_for_dp_per_m[m]) == len(bwd_flops_for_dp_per_m[m])
                    for k in range(len(fwd_flops_for_dp_per_m[m])):
                        split_num = fwd_flops_for_dp_per_m[m][k][2]
                        split_idx = fwd_flops_for_dp_per_m[m][k][3]
                        fwd_cost = fwd_flops_for_dp_per_m[m][k][0]
                        bwd_cost = bwd_flops_for_dp_per_m[m][k][0]

                        forward_cost_cmp.append([fwd_cost])
                        backward_cost_cmp.append([bwd_cost])

                    max_iter_sum_among_dp = simulate_time(forward_cost_cmp, backward_cost_cmp, PP, VPP)
                    
                    max_iter_sum_among_dp_list.append(max_iter_sum_among_dp)
                    max_sum_per_iter = max(max_sum_per_iter, max_iter_sum_among_dp)

                    if config.run_memory_simulator:
                        peak_memory = simulate_memory(seq_len_for_dp, config)

                    forward_cost_cmp = torch.tensor(forward_cost_cmp).flatten().tolist()
                    backward_cost_cmp = torch.tensor(backward_cost_cmp).flatten().tolist()

                    fwd_cost_total = sum(forward_cost_cmp)
                    bwd_cost_total = sum(backward_cost_cmp)

                    fwd_bwd_cost_total = fwd_cost_total + bwd_cost_total
                    num_microbatch = (i+1) * PP
                    pp_bubble_ratio = compute_pp_bubble_ratio(PP, num_microbatch, VPP)

                    pp_bubble_time = fwd_bwd_cost_total / (1 - pp_bubble_ratio) - fwd_bwd_cost_total
                    bubble_idle_time = max_iter_sum_among_dp - fwd_bwd_cost_total
                    imbalanced_bubble_time = bubble_idle_time - pp_bubble_time

                    bubble_over_iter_time = bubble_idle_time / max_iter_sum_among_dp
                    bubble_over_compute_time = bubble_idle_time / fwd_bwd_cost_total

                    pp_bubble_over_iter_time = pp_bubble_time / max_iter_sum_among_dp
                    pp_bubble_over_compute_time = pp_bubble_time / fwd_bwd_cost_total

                    imbalanced_bubble_over_iter_time = imbalanced_bubble_time / max_iter_sum_among_dp
                    imbalanced_bubble_over_compute_time = imbalanced_bubble_time / fwd_bwd_cost_total

                    bubble_time_list.append({
                        "pp_bubble_ratio": pp_bubble_ratio,
                        "bubble_over_compute_time":bubble_over_compute_time,
                        "pp_bubble_over_compute_time":pp_bubble_over_compute_time,
                        "imbalanced_bubble_over_compute_time":imbalanced_bubble_over_compute_time,
                    })

                    if config.run_memory_simulator and peak_memory >= 70 * 1024**3:
                        max_sum_per_iter = sys.float_info.max / 10.0    # skip this m
                        print(f"rank={torch.distributed.get_rank()}, Peak memory usage: {peak_memory / 1024**3:.2f} GiB, {combination=}")

                if torch.distributed.get_rank() == 0:
                    print(f"{combination=}")
                    for k in range(len(bubble_time_list)):
                        for key in bubble_time_list[k].keys():
                            bubble_time_list[k][key] = round(bubble_time_list[k][key], 3)
                        print(f"{k=}, {bubble_time_list[k]}")

        max_max_iter_sum = max(max_iter_sum_among_dp_list)
        min_max_iter_sum = min(max_iter_sum_among_dp_list)
        sum_max_iter_sum = sum(max_iter_sum_among_dp_list)
        len_max_iter_sum = len(max_iter_sum_among_dp_list)
        mean_max_iter_sum = sum_max_iter_sum/len_max_iter_sum

        # print(f"{sample_ids_2d=}")

        return indices_buckets_2d, sample_ids_2d, max_sum_per_iter, max_seq_per_m, used_flops


def simulate_memory(chunks_list, config):
    from megatron.pipeline_simulator.hotsim.model import Model
    from megatron.pipeline_simulator.hotsim.memory_model import MemoryModel
    from megatron.pipeline_simulator.hotsim.training_config import TrainingConfig
    from megatron.pipeline_simulator.hotsim.schedule import build_splitfuse_schedule
    model = Model(
        name="Llama",
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.ffn_hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
    )
    ckpt_type = "no"
    if config.recompute_granularity == "full":
        ckpt_type = "full"
    # if config.kaimm_recompute_mlp_activation_func and config.kaimm_recompute_norm:
    #     if config.kaimm_recompute_mlp_fc1:
    #         ckpt_type = "partial+fc1"
    #     else:
    #         ckpt_type = "partial"

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        num_gpus = torch.distributed.get_world_size()
    else:
        num_gpus = parallel_state.get_tensor_model_parallel_world_size() \
             * parallel_state.get_pipeline_model_parallel_world_size() \
             * parallel_state.get_data_parallel_world_size()
    train_config = TrainingConfig(
        model=model,
        num_gpus=num_gpus,
        microbatch_size=1,
        tensor_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        context_parallel_size=parallel_state.get_context_parallel_world_size(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        pipeline_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        expert_parallel_size=parallel_state.get_expert_model_parallel_world_size(),
        num_model_chunks=1,
        ckpt=ckpt_type,
        offload_ratio=0,
        # offload_ratio=config.kaimm_offload_activation_ratio,
    )

    actions_by_rank = build_splitfuse_schedule(
        config.pipeline_model_parallel_size, chunks_list
    )

    memory_model = MemoryModel(train_config)
    memory_model.setup(chunks_list, actions_by_rank)
    memory_model.run()
    return max(memory_model.peak_memory_histogram)


def simulate_time(fwd_costs, bwd_costs, PP, VPP):
    # PP = mpu.get_pipeline_model_parallel_world_size()
    schedule = SplitFuseSchedule(PP, fwd_costs, bwd_costs)
    # num_VPP = 8
    # schedule = InterleavedSchedule(PP, num_VPP, fwd_costs, bwd_costs)
    return test_with_schedule(schedule)
