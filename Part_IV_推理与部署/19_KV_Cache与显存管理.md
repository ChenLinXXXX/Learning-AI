# 19 KV Cache 与显存管理(PagedAttention / KV 量化 / Prefix Cache / 压缩)

> **章节定位**: Part IV 第 2 章,LLM 推理的"内存系统"
> **预计阅读时间**: 60-90 分钟
> **难度**: ★★★★☆
> **前置知识**: §07 Self-Attention、§18 推理流程
> **后续依赖**: §21 推理优化、§22 部署、§24 MLA

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [19.1 KV Cache 是什么、为什么需要](#191-kv-cache-是什么为什么需要)
- [19.2 KV Cache 显存账](#192-kv-cache-显存账)
- [19.3 朴素布局:Contiguous Tensor 的痛](#193-朴素布局contiguous-tensor-的痛)
- [19.4 PagedAttention:分页 KV cache](#194-pagedattention分页-kv-cache)
- [19.5 KV 量化:INT8 / INT4 / FP8](#195-kv-量化int8--int4--fp8)
- [19.6 KV 压缩:MQA / GQA / MLA](#196-kv-压缩mqa--gqa--mla)
- [19.7 Prefix Cache 与 RadixAttention](#197-prefix-cache-与-radixattention)
- [19.8 KV Cache Offloading 与跨节点传输](#198-kv-cache-offloading-与跨节点传输)
- [19.9 完整代码:KV cache + 简单 paged 实现](#199-完整代码kv-cache--简单-paged-实现)
- [19.10 速查:KV 大小估算表](#1910-速查kv-大小估算表)
- [🎨 19.10.A 通俗类比合集](#-1910a-通俗类比合集)
- [19.11 常见疑问](#1911-常见疑问)
- [19.12 本章小结](#1912-本章小结)
- [19.13 延伸阅读](#1913-延伸阅读)
- [19.14 实战练习](#1914-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

KV cache 是 LLM 推理的"内存系统",其大小、布局、压缩方式直接决定:**单 GPU 能服务多少并发、多长上下文、多大模型**。本章按"问题 → 工程方案"展开:

- KV cache 显存账:为什么 128k 上下文 / 70B MHA 模型一句话能吃 30 GB
- PagedAttention(vLLM):把 cache 切页消除内部碎片,Continuous Batching 的前置
- KV 量化:INT8 / INT4 / FP8,几乎无损 4-8× 压缩
- 结构性压缩:MQA、GQA、**MLA**(DeepSeek 的"必杀技",§24 深入)
- Prefix Cache / RadixAttention:多轮对话共享前缀的 cache
- Offloading 与跨节点 KV 传输

读完你能解释 vLLM 为什么是事实标准、MLA 为什么是 DeepSeek-V3 的核心创新。

---

## 学习目标

完成本章后,你能够:

- ✅ 估算给定模型、上下文长度、并发数的 KV cache 显存 (公式 19.1)
- ✅ 解释 PagedAttention 的 page table 与"零碎片"机制 (§19.4)
- ✅ 区分 MHA / MQA / GQA / MLA 的 KV cache 大小 (§19.6 表)
- ✅ 解释 Prefix Cache 为何在 RAG / agent 场景收益巨大 (§19.7)
- ✅ 推算 KV INT4 量化的精度损失级别 (§19.5)

---

## 19.1 KV Cache 是什么、为什么需要

### 19.1.1 朴素 attention 在 decode 时的浪费

decode 阶段每生成一个新 token,attention 需要的全部"历史 K, V" 与上一步相同。如果每步都重算:**$O(n^2)$ 退化成 $O(n^3)$**。

### 19.1.2 KV cache 的核心

把每一层、每一位置的 K 和 V 算出后**持久化**,后续 decode 时只需 append 新 token 的 K, V,无须重算历史:

$$
\text{cache}^{(l)} = (K^{(l)}_1, V^{(l)}_1), (K^{(l)}_2, V^{(l)}_2), \ldots
$$

decode 第 $t$ 步:

```text
q_t = W_Q · h_t         (只算当前 token)
k_t = W_K · h_t
v_t = W_V · h_t
append (k_t, v_t) 到 cache
attn = softmax(q_t · K_cache^T / √d) · V_cache    (历史 K, V 直接取)
```

复杂度变 **$O(n d)$ per step**,总 $O(n^2 d)$,与 prefill 同级。

### 19.1.3 注意 Q 不缓存

Q 与当前 token 强绑定(每步是不同 query)→ 缓存无意义。只缓存 K, V。

---

## 19.2 KV Cache 显存账

### 19.2.1 单序列单层

一层的 K cache 形状:$(n, h_{\text{kv}}, d_k)$,V 同。

每个元素 bytes_per_elem 字节(bf16 = 2,fp16 = 2,int8 = 1,int4 = 0.5)。

单序列 KV cache 总大小:

$$
\text{KV}_{\text{single}} = 2 \cdot L \cdot n \cdot h_{\text{kv}} \cdot d_k \cdot \text{bytes} \tag{19.1}
$$

其中 $L$ 是层数,因子 2 是 K 与 V。

### 19.2.2 LLaMA-2-7B 实例

参数:$L=32$,$h_{\text{kv}}=32$(MHA),$d_k=128$,bf16 = 2 bytes。

- $n = 1k$:$2 \cdot 32 \cdot 1024 \cdot 32 \cdot 128 \cdot 2 = 512$ MB
- $n = 4k$:2 GB
- $n = 32k$:**16 GB**
- $n = 128k$:**64 GB** ← 超过单 H100 80GB,根本装不下!

这就是为什么"长上下文 LLaMA-2"基本没法在单卡跑——KV cache 比权重还大。

### 19.2.3 LLaMA-3-70B(GQA $h_{\text{kv}}=8$)

KV 字节数 ÷ $32 / 8 = 4$ → 上述数字 ÷ 4。

### 19.2.4 DeepSeek-V3(MLA)

KV 压到约 $4 r$ 字节/层/token($r \approx 4 d_k$)→ 单 token KV 仅约 70 字节(vs MHA 32×128×2×2 ≈ 16 KB)→ **压缩 ~200×**!详见 §24。

### 19.2.5 并发与上下文的二维成本

| 模型 | n=4k | n=32k | n=128k | 多 32 并发 4k |
| --- | --- | --- | --- | --- |
| LLaMA-2-7B(MHA) | 2 GB | 16 GB | 64 GB | 64 GB |
| LLaMA-3-70B(GQA) | 1.25 GB | 10 GB | 40 GB | 40 GB |
| DeepSeek-V3(MLA) | 0.03 GB | 0.24 GB | 1 GB | 1 GB |

→ KV 压缩本身就是商业模型经济学的核心变量。

---

## 19.3 朴素布局:Contiguous Tensor 的痛

### 19.3.1 朴素实现

为每个请求预分配一个大小为 (L, max_len, h_kv, d_k) 的连续张量。

### 19.3.2 三大问题

1. **内部碎片(internal fragmentation)**:为最坏情况(max_len)分配,实际只用一半 → 半数显存浪费
2. **外部碎片**:不同长度请求来回腾位,可用显存被切碎
3. **不支持动态加入**:Continuous Batching(§18.5)无法工作

### 19.3.3 显存利用率惨象

实测:朴素布局 KV 利用率常在 **20-40%**,意味着 60-80% 显存浪费在 padding。

---

## 19.4 PagedAttention:分页 KV cache

### 19.4.1 灵感来自虚拟内存

把 KV cache 切成固定大小的**页(page / block)**,典型 16 token / page。每个序列维护一张**页表(block table)**,逻辑序号 → 物理页号。

物理页是显存里一个 pool 中的连续小块,**所有请求共享**。

### 19.4.2 attention 计算

PagedAttention 的 kernel **直接按页表读取**,不要求物理 K/V 在显存里连续。等价数学,但 IO pattern 不同——需要专门 CUDA kernel(vLLM 提供)。

### 19.4.3 收益

| 项 | 朴素 | PagedAttention |
| --- | --- | --- |
| 显存利用率 | 20-40% | **> 90%** |
| 单卡并发数 | 几 | **几十** |
| 支持 Continuous Batching | ❌ | ✅ |
| 支持 prefix sharing | ❌ | ✅(同物理页 + 引用计数) |

### 19.4.4 工程要点

- 页大小:典型 16-32。太小 → 元数据爆炸;太大 → 碎片回头
- 抢占(preemption):cache 满了驱逐低优请求(把它的页释放),被驱逐请求**重新 prefill**
- swap to CPU:可选,把页移到 CPU memory,需要时再 swap 回(慢但能拯救 OOM)

### 19.4.5 影响

PagedAttention 是 vLLM 的核心创新,2023 年 SOSP 论文发表后,**几乎所有主流推理引擎都已实现等价机制**(SGLang、TensorRT-LLM、TGI 等)。

---

## 19.5 KV 量化:INT8 / INT4 / FP8

### 19.5.1 思路

KV cache 是矩阵,本质上和权重量化类似:**把 fp16 / bf16 的 K, V 用更低 bit 存储**,attention 计算时 dequantize。

### 19.5.2 量化方案

| 方案 | bits | 压缩 | 精度损失 |
| --- | --- | --- | --- |
| bf16(基线) | 16 | 1× | 0 |
| FP8(E4M3 / E5M2) | 8 | 2× | < 0.1 PPL |
| INT8 per-token | 8 | 2× | < 0.1 PPL |
| INT4 per-token | 4 | 4× | 0.1-0.5 PPL(可接受) |
| INT2(实验) | 2 | 8× | 显著退化 |

### 19.5.3 量化粒度

- **per-tensor**:整个 K 一组 scale → 简单但精度差
- **per-token**:每 token 一个 scale → 平衡(主流)
- **per-channel**:每 d_k 维一个 scale → 精度高但开销大

vLLM、TensorRT-LLM 都支持 INT8 / FP8 KV cache,代码层只是一行配置。

### 19.5.4 与权重量化对照

权重量化:静态(模型加载时一次)。KV 量化:**动态**(每 step 新生成的 K, V 都要量化)。所以 KV 量化的开销主要在 quantize/dequantize 的 kernel。

---

## 19.6 KV 压缩:MQA / GQA / MLA

回到 §07.9 的 attention 变种,从 KV cache 视角再看一遍:

| 变种 | $h_{\text{kv}}$ | KV 字节(LLaMA-2-7B,1 token,1 层,bf16) | KV 总大小(4k) |
| --- | --- | --- | --- |
| **MHA** | $h = 32$ | $2 \cdot 32 \cdot 128 \cdot 2 = 16$ KB | 2 GB |
| **MQA** | 1 | 0.5 KB | 64 MB |
| **GQA** | 8(LLaMA-3) | 4 KB | 512 MB |
| **MLA** | "潜变量 $r$ + 解耦 RoPE" | ~70 B | ~10 MB |

### 19.6.1 MQA(Multi-Query Attention)

所有 head 共享 K, V(只 1 套 K/V)→ KV 压 head 倍。质量略损但可调,Falcon、PaLM 等用过。

### 19.6.2 GQA(Grouped-Query Attention)

折中:把 head 分成 $g$ 组,组内共享 K/V → KV 压 head/g 倍。LLaMA-2-70B / LLaMA-3 全系列、Mistral、Qwen 都用 GQA。

### 19.6.3 MLA(Multi-head Latent Attention)

DeepSeek 的"杀招":把 KV 压成低秩潜变量 $c \in \mathbb{R}^r$,每个 head 的 K, V 由 $c$ 通过两个上投影还原。结合 decoupled RoPE 解决位置编码与潜变量的兼容性。详见 §24。

### 19.6.4 选型指南

| 场景 | 选 |
| --- | --- |
| 学术 / 旧代码 | MHA |
| 中等模型 | GQA |
| 长上下文 / 大并发 / 商业服务 | **MLA** |
| 极端长度(RAG、Agent) | MLA + INT4 量化 |

---

## 19.7 Prefix Cache 与 RadixAttention

### 19.7.1 动机

多轮对话、RAG、Agent 中,**多个请求共享前缀**(system prompt、文档上下文、对话历史)。Prefill 同一份前缀几十次极浪费。

### 19.7.2 Prefix Cache 实现

把 prefill 算出的 KV cache **按前缀哈希缓存**,新请求带相同前缀时直接复用:

```text
请求 A: "<system> ... <user> 你好"           → prefill 全部
请求 B: "<system> ... <user> 今天天气"        → 复用 <system>...部分的 KV
```

收益:命中率高(50-90% 场景)时,prefill 算力降数倍。

### 19.7.3 RadixAttention(SGLang)

SGLang 把 prefix tree 抽象成 **Radix Tree(基数树)**:任何前缀都能被精确匹配 + LRU 驱逐 + 跨请求共享。

工业实测:多轮对话场景吞吐 2-5×。SGLang 因此在 chatbot 服务领域强势崛起。

### 19.7.4 与 PagedAttention 协同

页表里的物理页带引用计数。多请求共享同前缀页,只在最后一个释放时回收。是 vLLM、SGLang 等共有的设计。

---

## 19.8 KV Cache Offloading 与跨节点传输

### 19.8.1 Offloading 到 CPU / NVMe

当 GPU KV cache 不够时,把"不活跃"的 cache 页移到:

- CPU memory(PCIe 带宽 ~ 30 GB/s)
- NVMe SSD(几 GB/s)

激活页移回 GPU 时延迟显著,但能支持远超 GPU 显存的"虚拟"上下文。

### 19.8.2 跨节点传输

DistServe / Mooncake 等架构在 prefill 节点算完 KV cache 后,**用 RDMA 把 cache 发送到 decode 节点**。

- 7B 模型 4k 上下文 ~ 2 GB → InfiniBand 200 Gbps ~ 80 ms
- 实际可重叠到 decode 第一步前,接近 0 延迟开销

### 19.8.3 KV cache 与系统设计

KV cache 已经不再是"模型层细节",而是**系统设计的一等公民**:

- 调度按 cache 命中调
- 路由按 cache 位置选节点
- 量化、压缩、shareing 在多个层次同时进行

DeepSeek-V3 推理系统、Kimi Mooncake、OpenAI 内部都把 cache 当独立的"分布式数据库"看待。

---

## 19.9 完整代码:KV cache + 简单 paged 实现

### 19.9.1 普通 cache 版 attention

```python
import torch
import torch.nn.functional as F

class CachedAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads; self.d_k = d_model // n_heads
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, past_kv=None):
        # x: (B, T, d) — T=1 in decode, T=n in prefill
        B, T, d = x.shape
        q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        if past_kv is not None:
            K_past, V_past = past_kv
            k = torch.cat([K_past, k], dim=2)   # 沿序列维拼
            v = torch.cat([V_past, v], dim=2)

        scores = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        # decode 时 T=1,causal mask 自然成立(只对历史 + 自己看)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return self.W_o(out), (k, v)
```

### 19.9.2 极简 paged 内存池

```python
class PagedKVPool:
    def __init__(self, n_pages, page_size, n_heads, d_k, dtype=torch.bfloat16):
        # 一整块连续显存,逻辑切成 N 页
        self.k = torch.zeros(n_pages, page_size, n_heads, d_k, dtype=dtype, device="cuda")
        self.v = torch.zeros_like(self.k)
        self.free = list(range(n_pages))

    def alloc(self):
        if not self.free: raise RuntimeError("KV OOM")
        return self.free.pop()

    def free_page(self, idx):
        self.free.append(idx)

class PagedSequence:
    """每个请求一份页表"""
    def __init__(self, pool: PagedKVPool, page_size: int):
        self.pool = pool
        self.page_size = page_size
        self.pages = []     # 物理页号列表
        self.length = 0

    def append(self, k_new, v_new):
        # k_new, v_new: (1, T_new, h, d_k)
        T = k_new.size(1)
        for i in range(T):
            if self.length % self.page_size == 0:
                self.pages.append(self.pool.alloc())
            p = self.pages[-1]
            slot = self.length % self.page_size
            self.pool.k[p, slot] = k_new[0, i]
            self.pool.v[p, slot] = v_new[0, i]
            self.length += 1

    def gather(self):
        # 把所有 token 的 K, V 拼回连续(只在算 attention 时调用)
        idx = torch.tensor(self.pages, device="cuda")
        K = self.pool.k[idx].view(-1, self.pool.k.size(2), self.pool.k.size(3))[:self.length]
        V = self.pool.v[idx].view(-1, self.pool.v.size(2), self.pool.v.size(3))[:self.length]
        return K.unsqueeze(0), V.unsqueeze(0)
```

> 教学版,真实 vLLM 把 `gather` 融合进 attention kernel,避免显存拷贝。

---

## 19.10 速查:KV 大小估算表

公式 (bf16):**`KV 字节 ≈ 2 · L · n · h_kv · d_k · 2`**

| 模型 | L | h_kv | d_k | 单 token KV | 32k 上下文 KV |
| --- | --- | --- | --- | --- | --- |
| GPT-2 small | 12 | 12 | 64 | 36 KB | 1.15 GB |
| LLaMA-2-7B | 32 | 32 | 128 | 512 KB | **16 GB** |
| LLaMA-2-70B | 80 | 64 | 128 | 2.6 MB | 80 GB |
| LLaMA-3-70B(GQA) | 80 | 8 | 128 | 0.33 MB | 10 GB |
| Mistral-7B(GQA) | 32 | 8 | 128 | 128 KB | 4 GB |
| DeepSeek-V3(MLA) | 61 | — | $r \approx 512$ | ~70 B | **~2 MB**(<1%) |

→ MLA 是 100-200× 压缩,使 671B 模型 128k 上下文单卡可服务多并发。

---

## 🎨 19.10.A 通俗类比合集

### 类比一: 写小说时的速记本 📓

> 模型边写边记速记本,免得每写一段都重读全文。

- **KV cache** = 速记本(已读 / 已写章节的关键信息)
- **PagedAttention** = 速记本不强制按顺序连续装订,**每页可独立**;桌面只放当下用的页
- **量化** = 速记简写,字数少但能看懂
- **MLA** = 把速记本压成一行索引,真要查时用"展开公式"还原
- **Prefix Cache** = 多人共写同一系列小说,**共享前几章的速记**

### 类比二: 餐厅厨房备料 🍳

> Prefill = 备齐基础配料;Decode = 一道道炒。

- **KV cache** = 备料盘(切好的葱、姜、蒜)
- **朴素布局** = 每位厨师都有自己的大托盘,常常空着 60%
- **PagedAttention** = 大家共享一片"备料区",按小格分发,谁需要拿谁的
- **Prefix Cache** = 多桌客人都点了酸辣汤底,**底料只熬一锅大家分**
- **KV 量化** = 配料用小份分装,体积小但味道几乎不变
- **MLA** = 配料压成调味粉,用时加水还原

### 类比三: 图书馆的"快速借阅区" 📚

> KV cache 是图书馆放在前台的"高频借阅副本"。

- **PagedAttention** = 前台书架按小格分类,书可以放不同格里
- **Offloading** = 前台放不下的书暂存到地下室(CPU/NVMe)
- **Prefix Cache** = 多人都要查同一本字典 → 只放一本前台共享
- **KV INT4** = 把书印成口袋本,字小但能读
- **MLA** = 字典内容做成精华摘要,真要细节再去主馆翻

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 📓 速记本 | **历史 / 增量** | KV cache 思想 |
| 🍳 备料 | **池化共享** | PagedAttention |
| 📚 借阅区 | **分级存储** | offloading / 量化 / MLA |

**记忆口诀**:
> **"KV 是历史,分页是池化,量化和 MLA 是压缩术;前缀复用,远水救近火。"**

---

## 19.11 常见疑问

### Q1: KV cache 比权重还大,这正常吗?
**A**: 对长上下文 + MHA 模型非常正常。LLaMA-2-7B 在 128k 上下文,KV (64 GB) 远超权重 (14 GB)。这就是为什么 LLaMA-3 用 GQA、DeepSeek 用 MLA。

### Q2: PagedAttention 不会变慢吗?
**A**: 略慢(随机访存模式)。但显存利用率提升带来 batch size 提升,总吞吐反而**显著提高**。vLLM 实测 throughput 2-10×。

### Q3: KV 量化用 INT4 安全吗?
**A**: 对短/中上下文(< 32k)与 chat 任务,~ 0.1-0.3 PPL 损失,人感觉不到。对极长上下文 / 高精度任务(代码、数学),建议 FP8 或 INT8。

### Q4: 推理时能切换 MHA → GQA / MLA 吗?
**A**: 不能。它们改变了模型结构,需要从训练时就用对应架构。LLaMA-2-7B 不能"运行时切到 GQA"。

### Q5: 我跑 Qwen-7B 32k,KV 装不下怎么办?
**A**:
1. 开启 KV INT8 / INT4
2. 用 vLLM(PagedAttention 提升利用率)
3. 缩短上下文
4. 换模型(GQA / MLA 友好的)
5. 多卡张量并行(TP)

---

## 19.12 本章小结

### 核心公式

$$
\text{KV cache size} = 2 \cdot L \cdot n \cdot h_{\text{kv}} \cdot d_k \cdot \text{bytes}
$$

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **KV cache** | decode 时间复杂度从 $O(n^3)$ 降到 $O(n^2 d)$ |
| **显存大头** | 长上下文时常超过权重 |
| **PagedAttention** | 消除碎片 + 支持 Continuous Batching |
| **量化** | INT8/INT4/FP8 几乎无损 4-8× 压缩 |
| **结构压缩** | MHA → GQA → MLA |
| **Prefix Cache** | 多请求共享前缀,RAG/Agent 必备 |

### 一句话总结

> **KV cache 是 LLM 推理的"内存系统",在长上下文与高并发场景里常常超过权重大小;PagedAttention 把它切成可共享的页消除碎片、解锁 Continuous Batching;INT8/INT4/FP8 量化 + MQA/GQA/MLA 结构性压缩共同把 KV 体积压回可接受范围——其中 MLA 是 DeepSeek-V3 在 128k 上下文 + 大并发下经济性的核心来源。**

---

## 19.13 延伸阅读

### 必读论文

1. Pope et al., 2023. *"Efficiently Scaling Transformer Inference"*. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)(KV cache 经典分析)
2. Kwon et al., 2023. *"Efficient Memory Management for LLM Serving with PagedAttention"*(vLLM)[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
3. Shazeer, 2019. *"Fast Transformer Decoding: One Write-Head is All You Need"*(MQA)[arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

### 进阶论文

4. Ainslie et al., 2023. *"GQA: Training Generalized Multi-Query Transformer Models"*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
5. DeepSeek-AI, 2024. *"DeepSeek-V2"*(MLA 提出)
6. Zheng et al., 2024. *"SGLang: Efficient Execution of Structured Language Model Programs"*(RadixAttention)[arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
7. Hooper et al., 2024. *"KVQuant: Towards 10 Million Context Length LLM Inference"*. [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)

### 推荐资源

- vLLM blog 系列:*"How continuous batching enables 23x throughput"*
- SGLang 文档:*"RadixAttention"*
- HuggingFace `transformers` 的 `Cache` 抽象类

---

## 19.14 实战练习

### 练习 1 (★)

写一个小脚本,输入 (L, h_kv, d_k, n, dtype),输出 KV cache 字节数;复现 §19.10 表。

### 练习 2 (★★)

把 §19.9.1 的 `CachedAttention` 套到 §10.6 的 GPTMini,实现 `forward(x, past_kvs)`,跑通生成。

### 练习 3 (★★)

用 vLLM 服务 Qwen-7B-Chat,分别开启 / 关闭 KV INT8 量化,在 32k prompt 下测试显存与 latency。

### 练习 4 (★★★)

实现一个最小 PrefixCache:hash(prompt前 N token)→ 复用已 prefill 的 KV cache。在重复 system prompt 场景验证 prefill 算力节省。

### 练习 5 (★★★)

读 vLLM `paged_attention_v2.cu` 实现,梳理 block_table 怎么传入 kernel、怎么按页索引 K/V;画一份数据流图。

---

## 章节交叉引用

- 前置: [§07 Self-Attention](../Part_II_Transformer架构/07_Self_Attention机制.md), [§18 推理流程](18_推理流程.md)
- 后续: [§21 推理优化](21_推理优化.md)(Flash Attention 与 KV cache 协同), [§22 本地部署](22_本地部署实战.md)
- 相关: [§24 MLA](../Part_V_DeepSeek专题/24_MLA深度解析.md)(KV 结构压缩的极致)

---

*下一章: §20 采样策略*
