# QSA compact：BF16 / FP8 用户指南

`atrex.qsa_compact` 在 NVIDIA SM100 / SM103 上提供分页稀疏 GQA attention，
直接接收四-token block IDs，并融合 sigmoid gate。BF16 与 FP8 共用接口；gate 和输出均为 BF16。
使用 CUDA 13+ 和 CuTeDSL 4.8.1；完整依赖与安装命令见测试说明。

| 架构 | Prefill / ragged batched prefill | Q1 decode / Q4 MTP | CUDA Graph |
|---|---|---|---|
| SM100 | BF16、FP8 E4M3FN | BF16、FP8 E4M3FN | 预热后的 Q1/Q4 支持 |
| SM103 | BF16、FP8 E4M3FN | BF16、FP8 E4M3FN | 预热后的 Q1/Q4 支持 |

## 调用与张量约定

```python
import atrex

out = atrex.qsa_compact(
    q, k_cache, v_cache, indexer_block_ids, query_positions,
    block_table, token_to_req, use_prefill_config, output_gate,
    out=None, sm_scale=None, v_scale=1.0,
)
```

| 参数 | dtype / shape | 约束 |
|---|---|---|
| `q` | BF16 / E4M3FN `[T,Hq,256]` | `T>0`，D 连续，base 和行/head stride 为 16B 对齐 |
| `k_cache`, `v_cache` | 与 Q 相同 dtype，`[P,S,Hkv,256]` | 相同 shape/stride；`P>0`，`S` 为 16 的正整数倍，`Hkv=1/2`；D 连续，base 和各维 stride 为 16B 对齐（D 维除外） |
| `indexer_block_ids` | int32 `[T,512]` | 连续、16B 对齐；请求内四-token 组编号 |
| `query_positions` | int32/int64 `[T]` | 连续；请求内的 token 位置 |
| `block_table` | int32 `[B,L]` | 连续；`B,L>0`，元素为物理页编号 |
| `token_to_req` | int32 `[T]` | 连续；每行 query 的请求归属，可交错排列 |
| `output_gate`, `out` | BF16 `[T,Hq,256]` | 连续、32B 对齐；out 不与任何输入共享 storage |

`Hq/Hkv` 为 8 或 12。所有张量位于同一设备，维度不重叠，仅支持推理。
K/V 可独立分配，也可为同一交错 pool 的两个 view。输入不被修改；传入 out 时返回
该张量，否则分配输出。`use_prefill_config` 仅用于调用兼容，实际路径由形状决定。

位置 `p>=0` 的一行消费前 `min(512,(p+1)//4)` 个完整组，并追加长度为 `(p+1)%4`
的当前组尾部。有效完整组须互异且位于当前组之前，调用者负责保证选择集合合法。
负 ID、无效请求或页映射被 mask；没有有效 token 的行输出零。不支持重复完整组或任意 token mask。
物理位置由 `block_table[request_id, local_token//S]` 和 `local_token%S` 确定，物理页可不连续。
页大小包括 16/32/48 等小页以及 1056/2112 等非 128 倍数，不需要补齐至 128 倍数。
四-token 组不会跨越页边界。Prefill 的一个 bitmap word 表示 128 token；
当 `S<128` 时，对每个选中组检查页表有效性，以正确处理一个 word 内的多个无效页。
`S>=128` 的未重排 bitmap 沿用首尾页掩码。框架 adapter 的页大小校验须与此约定一致。
输出按 `BF16(float(BF16(attention)) * sigmoid(float(gate)))` 计算。

## FP8 scale

调用者先将 Q/K/V 量化为 `torch.float8_e4m3fn`，并按反量化乘数约定传参：

```python
# Q_real = Q_fp8 * q_scale，K/V 同理
out = atrex.qsa_compact(
    q, k_cache, v_cache, indexer_block_ids, query_positions,
    block_table, token_to_req, use_prefill_config, output_gate,
    sm_scale=(q.shape[-1] ** -0.5) * q_scale * k_scale,
    v_scale=v_scale,
)
```

`sm_scale` 是完整 QK 乘数，已包含 `1/sqrt(D)`，默认 `None` 等价于 `1/sqrt(D)`；
`v_scale` 默认为 1，在 attention 的 BF16 舍入前应用。二者为有限、正、可用 FP32
表示的 Python 实数，不接受 bool 或 tensor。每次调用仅支持一组全局标量；
不提供 per-head/per-channel scale。BF16 输入须使用默认 scale 参数。
接口不负责量化、校准或写入 KV cache。

| 调用者持有的 scale | `sm_scale` | `v_scale` |
|---|---|---|
| PrimTS | `sm_scale` 原值 | `v_scale` 原值 |
| Triton QSA | `bmm1_scale` | `bmm2_scale` |
| 独立 Q/K/V 反量化乘数 | `D**-0.5 * q_scale * k_scale` | V 反量化乘数 |

## Batch、KV 容量与 CUDA Graph

| 执行路径（两种架构、dtype 相同） | 形状 | CUDA Graph |
|---|---|---|
| 专用 Q1 decode / Q4 MTP | `B<=256` 且 `T=B` 或 `T=4B` | 预热后支持，须提供 out |
| 通用 prefill | 其余形状，包括 `B>256` | 不支持 capture |

请求归属始终取自 `token_to_req`，不要求各请求拥有相同数量的 query。
Prefill 使用局部 bitmap，没有固定 batch 或总 query 数上限；每请求页表容量要求
`L <= ceil(1,048,576/S)`。page=1056 的尾页因此可预留 1,048,608 token。
所有路径要求 `B*L*S<2^31`，query/workspace 索引不溢出 int32；实际分配还受显存限制。

首次调用按形状、布局和设备 JIT 编译；导入包不编译。无需调用者管理 workspace 或 plan。
Prefill 在库内缓存通过校验的静态张量签名及提交配置（各至多 128 项），
配置包含编译入口、标量和 scratch 尺寸，不持有输入、输出或 scratch 张量。
每次调用仍检查对齐、梯度属性和输出别名，并绑定当次张量及当前 CUDA stream。
四块 prefill scratch 来自一个调用内 arena；不同调用不共享可写 workspace。
新 shape 或新 scale 会重新建立提交配置，公开单函数接口和框架接入方式不变。
下面的 capture 示例适用于两种架构上满足专用 Q1/Q4 条件的输入：

```python
import torch

out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
# BF16 使用 {}；FP8 使用上文计算的 sm_scale 和 v_scale。
scale_args = {} if q.dtype == torch.bfloat16 else dict(sm_scale=sm_scale, v_scale=v_scale)
args = (q, k_cache, v_cache, indexer_block_ids, query_positions,
        block_table, token_to_req, False, output_gate, out)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    atrex.qsa_compact(*args, **scale_args)
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    atrex.qsa_compact(*args, **scale_args)
graph.replay()
```

Graph 固定 GPU 操作、地址和标量参数。输入、输出和 KV 存储须保持存活且地址稳定；
内容可以原地更新。更换 shape、布局或 FP8 scale 值需要重新 capture。
调用者负责跨 stream 依赖；共享 graph pool 时遵守 PyTorch 的执行顺序要求。
完整 vLLM backend 接入和模型端到端吞吐尚未验证。

## 实现与静态 dispatch

Prefill 和 decode/MTP 分别按架构与 dtype 选择实现。公共 BF16 decode、
容量检查、scale 和 host helper 位于 `common_utils`。
Decode 的规则优先级为：serving/148-SM 专用规则、标准布局弱区规则、
FP8 Q4 区间表、架构默认公式。规则仅读取张量元信息，不读取 GPU 长度或在线测速。
B 表示包含 padding 的分配槽位；未命中精细规则的输入继续使用通用配置。

Serving 专用规则覆盖 Q4、8:1、page128/1056/2112、148/152 SM，
K/V 元素 stride 为 `(18*S*512,512,512,1)`，其余张量连续且满足对齐要求。
其他精细规则限定标准连续布局与 152 SM。具体边界以
[`decode_dispatch.py`](../common_utils/decode_dispatch.py) 和
[`decode_intervals.py`](../common_utils/decode_intervals.py) 为准。
规则不保证任意 KV 分布都达到逐 case 最优；实测范围见
[测试说明](../../../../op_test/nvidia/qsa/README.md)。

## 精度与已知限制

参考使用同输入的 FP32 QK/softmax/PV，并按上述接口约定进行 BF16 舍入。
BF16 严格逐元素标准为 `abs(actual-ref)<=0.003+0.02*abs(ref)`，
但仅有 2/3 个可见 KV token 的抵消样本可能超差。
FP8 概率量化和归约引入额外误差，部分输入不满足该标准。
测试同时记录 relative L2、最大绝对误差、超差元素数及基线配对差异。
Q4 不保证逐位确定性。精度结论限已测分布，不包含输入 FP8 量化损失或模型端到端质量。

## 测试与复现

[test_qsa_sm100.py](../../../../op_test/nvidia/qsa/test_qsa_sm100.py) 和
[test_qsa_sm103.py](../../../../op_test/nvidia/qsa/test_qsa_sm103.py) 分别覆盖对应架构的
BF16/FP8 prefill、decode/MTP、页表与布局边界、跨 stream 及 Q1/Q4 Graph。
环境准备与基本回归命令见 [测试说明](../../../../op_test/nvidia/qsa/README.md)。
