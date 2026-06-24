# 07 Self-Attention 机制

> **章节定位**: Part II 核心章节,Transformer 的心脏
> **预计阅读时间**: 60-90 分钟
> **难度**: ★★★★☆
> **前置知识**: §01 线性代数(矩阵乘法、内积、softmax), §02 概率基础
> **后续依赖**: §08(FFN)、§09(归一化)、§10(Block 整体)、§19(KV Cache)、§24(MLA)

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [7.1 历史背景与动机](#71-历史背景与动机)
- [7.2 直觉:加权求和的图书馆比喻](#72-直觉加权求和的图书馆比喻)
- [7.3 Scaled Dot-Product Attention(标准形式)](#73-scaled-dot-product-attention标准形式)
- [7.4 Q、K、V 怎么来?](#74-qkv-怎么来-1)
- [7.5 Causal Mask(因果掩码)](#75-causal-mask因果掩码)
- [7.6 Multi-Head Attention](#76-multi-head-attention)
- [7.7 完整代码实现](#77-完整代码实现)
- [7.8 复杂度分析](#78-复杂度分析)
- [7.9 与其他变种的对比](#79-与其他变种的对比)
- [🎨 7.9.A 通俗类比合集](#-79a-通俗类比合集)
- [7.10 常见疑问](#710-常见疑问)
- [7.11 本章小结](#711-本章小结)
- [7.12 延伸阅读](#712-延伸阅读)
- [7.13 实战练习](#713-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

Self-Attention(自注意力)是 Transformer 架构的核心机制。它使每个 token 能够**直接**与序列中任意位置的其他 token 交互,无需经过中间的循环或卷积。本章从基础的"加权求和"直觉出发,逐步引出 Scaled Dot-Product Attention 的标准形式,推导关键的数学性质(包括 $\sqrt{d_k}$ 缩放的必要性、Softmax 的几何含义、因果掩码的实现),并在最后扩展到 Multi-Head Attention。

本章的核心目标:
1. **推导**: 给出 Self-Attention 完整的数学定义
2. **解释**: 每一步操作背后的几何与概率含义
3. **代码**: 从零实现一个可运行的 attention 层
4. **延伸**: 为后续 MLA、KV Cache、Flash Attention 等高级话题打基础

---

## 学习目标

完成本章后,你能够:

- ✅ 写出 Scaled Dot-Product Attention 的完整公式 (公式 7.4)
- ✅ 解释为什么必须除以 $\sqrt{d_k}$ (§7.3.2)
- ✅ 推导 Multi-Head Attention 的矩阵形式 (公式 7.7)
- ✅ 解释 Causal Mask 的工程实现 (§7.5)
- ✅ 从零写一个 30 行 PyTorch 的 attention 层 (§7.7)
- ✅ 分析 Attention 的计算复杂度与显存瓶颈 (§7.6)

---

## 7.1 历史背景与动机

### 7.1.1 前 Attention 时代

在 Transformer 出现之前(2017 年之前),序列建模的主流方案是 RNN/LSTM:

$$
h_t = f(h_{t-1}, x_t)  \tag{7.1}
$$

每个时刻的隐藏状态 $h_t$ 由上一时刻 $h_{t-1}$ 和当前输入 $x_t$ 计算得到。这种**串行依赖**带来两个严重问题:

1. **无法并行**: 必须按时间步逐个计算,GPU 利用率低
2. **长程依赖衰减**: 信息经过多次非线性变换后,远距离 token 之间的关联会被"稀释"

### 7.1.2 Attention 的引入(2015-2017)

Bahdanau 等人(2015)在机器翻译中首次提出 Attention 机制,作为 RNN 的辅助。核心思想:

> **解码时不依赖单一压缩向量,而是动态地"回看"编码器的每个位置。**

2017 年 Vaswani 等人发表 *"Attention is All You Need"*,提出**完全抛弃 RNN,只用 Attention** 的 Transformer 架构。这成为 LLM 时代的奠基性工作。

### 7.1.3 Self-Attention vs Cross-Attention

| 类型 | Q 来源 | K, V 来源 | 用途 |
|---|---|---|---|
| **Self-Attention** | 同一序列 | 同一序列 | 序列内部交互 |
| **Cross-Attention** | 序列 A | 序列 B | 跨序列交互(如 Decoder 看 Encoder) |

本章主要讨论 Self-Attention(LLM 中绝大多数 attention 都是这种)。

---

## 7.2 直觉:加权求和的图书馆比喻

在跳进公式之前,先用一个比喻建立直觉。

### 7.2.1 场景

你在图书馆查资料,要写一篇关于"苹果"的文章:

- **Q (Query, 查询)**: 你脑中的问题——"苹果是什么"
- **K (Key, 键)**: 每本书脊上的标签
- **V (Value, 值)**: 每本书的实际内容

你的工作流程:
1. 把你的 Query 与每本书的 Key 比较,得到**相关度分数**
2. 把分数归一化,得到**注意力权重**(权重和为 1)
3. 按权重**加权求和**所有书的 Value,作为最终答案

### 7.2.2 数学形式

设有 $n$ 本书,Q 是 1 个 query 向量,K、V 是 $n$ 个向量矩阵:

$$
\text{Attention}(Q, K, V) = \sum_{i=1}^{n} \alpha_i V_i  \tag{7.2}
$$

其中权重 $\alpha_i$ 由 Q 与 $K_i$ 的相似度决定:

$$
\alpha_i = \frac{\exp(\text{sim}(Q, K_i))}{\sum_{j=1}^{n} \exp(\text{sim}(Q, K_j))}  \tag{7.3}
$$

这就是 Attention 的本质——**基于相似度的加权平均**。

### 7.2.3 Self-Attention 的特殊性

在 Self-Attention 中,Q、K、V **都来自同一序列**:

```
输入序列: [我, 喜欢, 吃, 苹果]
            ↓        ↓        ↓
对每个 token,生成对应的 Q、K、V 向量
            ↓
每个 token 的 Q 与所有 token 的 K 比较
            ↓
得到 token-to-token 的注意力权重
            ↓
按权重融合所有 token 的 V
```

例如 "苹果" 这个 token 会:
- 用自己的 Q 去查询所有 token 的 K
- 发现 "吃" 的 K 与自己的 Q 高度相关
- 因此 "苹果" 的输出向量会**强烈融合** "吃" 的信息

这就是"上下文化表示"(*contextualized representation*)的来源。

---

## 7.3 Scaled Dot-Product Attention(标准形式)

### 7.3.1 公式定义

**定义 7.1** (Scaled Dot-Product Attention):

给定 query 矩阵 $Q \in \mathbb{R}^{n \times d_k}$,key 矩阵 $K \in \mathbb{R}^{n \times d_k}$,value 矩阵 $V \in \mathbb{R}^{n \times d_v}$,Scaled Dot-Product Attention 定义为:

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V}  \tag{7.4}
$$

其中:
- $n$: 序列长度
- $d_k$: query/key 的维度
- $d_v$: value 的维度(通常 $d_v = d_k$)
- $QK^\top \in \mathbb{R}^{n \times n}$: attention 分数矩阵
- $\frac{1}{\sqrt{d_k}}$: 缩放因子

公式拆解为四步:

| Step | 操作 | Shape | 含义 |
|---|---|---|---|
| 1 | $S = QK^\top$ | $[n, n]$ | 相似度分数 |
| 2 | $S' = S / \sqrt{d_k}$ | $[n, n]$ | 缩放(防梯度消失) |
| 3 | $A = \text{softmax}(S')$ | $[n, n]$ | 归一化为概率 |
| 4 | $O = AV$ | $[n, d_v]$ | 加权求和 |

### 7.3.2 为什么除以 $\sqrt{d_k}$?

这是最经典的考察点。

**问题**: 不除会怎样?

**分析**: 假设 $Q$ 和 $K$ 的元素独立、零均值、单位方差。则点积:

$$
(QK^\top)_{ij} = \sum_{l=1}^{d_k} Q_{il} K_{jl}  \tag{7.5}
$$

每一项 $Q_{il} K_{jl}$ 是零均值、单位方差的乘积(均值为 0,方差为 1)。因此:

$$
\text{Var}\left[(QK^\top)_{ij}\right] = d_k  \tag{7.6}
$$

**结论**: 点积的方差与 $d_k$ 成正比。当 $d_k$ 较大(如 64、128)时,点积的数值会很大(标准差为 $\sqrt{d_k}$)。

**后果**: 大数值进入 softmax 会导致**饱和**——最大值的概率接近 1,其他接近 0:

```
不缩放:  scores = [3√d_k, 1√d_k, 0.5√d_k]   → softmax 接近 one-hot
缩放后:  scores = [3, 1, 0.5]                → softmax 平滑分布
```

饱和的 softmax 梯度近乎 0,模型**无法学习**。

**解决**: 除以 $\sqrt{d_k}$,让方差回到 1:

$$
\text{Var}\left[\frac{(QK^\top)_{ij}}{\sqrt{d_k}}\right] = 1
$$

这就是 $\sqrt{d_k}$ 的来源——**数值稳定性的保证**。

### 7.3.3 Softmax 的几何含义

Softmax 把任意实数向量映射为概率分布:

$$
\text{softmax}(s)_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}, \quad \sum_i \text{softmax}(s)_i = 1  \tag{7.7}
$$

几何上,它把 $\mathbb{R}^n$ 投影到 $(n-1)$ 维单纯形(simplex)上。

**性质**:
- 单调性: $s_i > s_j \Rightarrow \text{softmax}(s)_i > \text{softmax}(s)_j$
- 平移不变: $\text{softmax}(s + c) = \text{softmax}(s)$
- 温度: $\text{softmax}(s / T)$,$T$ 越小越尖锐

在 attention 中,softmax 把"相似度分数"转换为"注意力权重",保证权重的可解释性(和为 1)和可微性。

### 7.3.4 公式 7.4 的矩阵化优势

为什么用矩阵形式而不是逐个计算?

**单 query 形式**:
```
for each query q_i:
    scores = [sim(q_i, k_j) for j in 1..n]
    weights = softmax(scores)
    output_i = sum(weights[j] * v_j)
```

**矩阵形式**:
```python
scores = Q @ K.T / sqrt(d_k)        # 一次性算完
weights = softmax(scores, dim=-1)
output = weights @ V
```

矩阵形式让所有 query 共享相同的计算,**GPU 利用率大幅提升**。这是 Transformer 能并行训练的关键。

---

## 7.4 Q、K、V 怎么来?

公式 7.4 假设 Q、K、V 已存在,但它们从哪里来?

### 7.4.1 线性投影

给定输入序列 $X \in \mathbb{R}^{n \times d_{\text{model}}}$(每行是一个 token 的隐藏向量),通过三个可学习的投影矩阵生成 Q、K、V:

$$
\begin{align}
Q &= X W_Q, \quad W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}  \tag{7.8} \\
K &= X W_K, \quad W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}  \tag{7.9} \\
V &= X W_V, \quad W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}  \tag{7.10}
\end{align}
$$

**直觉**:
- $W_Q$ 学习"如何提问"
- $W_K$ 学习"如何被检索"
- $W_V$ 学习"提供什么内容"

三个矩阵都是**独立训练**的,即使 Q 和 K 在公式上对称,它们的语义角色完全不同。

### 7.4.2 为什么不直接用 $X$ 作为 Q、K、V?

**回答**: 自由度不足。

如果 $Q = K = V = X$,则 attention 退化为:

$$
\text{softmax}(XX^\top / \sqrt{d}) X
$$

这只能表达"与自己相似的 token 互相聚合",无法表达更复杂的关系(如"修饰词找名词"、"动词找宾语")。

**通过引入** $W_Q$、$W_K$、$W_V$,模型可以学习不同的"投影方向",在投影后的空间里寻找有用的关系。

### 7.4.3 投影矩阵的维度选择

设隐藏维度为 $d_{\text{model}}$,头数为 $h$:

$$
d_k = d_v = \frac{d_{\text{model}}}{h}  \tag{7.11}
$$

例如 LLaMA-7B 中 $d_{\text{model}} = 4096$,$h = 32$,所以 $d_k = 128$。

**为什么这样选?** 让多头总维度等于单头维度,**总参数量不变**,但表达多样性提升。

---

## 7.5 Causal Mask(因果掩码)

### 7.5.1 问题

在**生成式**任务(如语言建模)中,第 $t$ 个 token 只应该看到 $1, 2, \ldots, t$,**不能看到** $t+1, t+2, \ldots$(避免"作弊")。

但公式 7.4 的标准 attention 让每个位置看到**所有**位置,违反因果性。

### 7.5.2 解决方案

在 softmax **之前**,对 $S' = QK^\top / \sqrt{d_k}$ 矩阵的"未来位置"加上 $-\infty$:

$$
S''_{ij} = \begin{cases}
S'_{ij} & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}  \tag{7.12}
$$

经过 softmax 后,$-\infty$ 的位置概率为 0:

$$
\text{softmax}(-\infty) = 0
$$

即"看不见"未来的 token。

### 7.5.3 矩阵化实现

构造一个**下三角矩阵**作为 mask:

$$
M = \begin{pmatrix}
0 & -\infty & -\infty & \cdots \\
0 & 0 & -\infty & \cdots \\
0 & 0 & 0 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$

然后:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V  \tag{7.13}
$$

PyTorch 实现:
```python
mask = torch.tril(torch.ones(n, n))                 # 下三角 1,上三角 0
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # [n, n]
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

### 7.5.4 Causal Mask vs Bidirectional

| 模型 | Mask | 用途 |
|---|---|---|
| **GPT / LLaMA / DeepSeek** (Decoder-only) | Causal | 生成式 |
| **BERT** (Encoder-only) | 无 mask | 双向理解 |
| **T5** (Encoder-Decoder) | Encoder 无 mask,Decoder 因果 | 翻译/摘要 |

当代大模型(GPT 系、LLaMA 系)**几乎全部是 Decoder-only**,因此都用 Causal Mask。

---

## 7.6 Multi-Head Attention

单头 attention 只能学习**一种**关系模式。Multi-Head Attention(MHA)让模型并行学习**多种**关系。

### 7.6.1 数学定义

**定义 7.2** (Multi-Head Attention):

$$
\boxed{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O}  \tag{7.14}
$$

其中每个头:

$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)  \tag{7.15}
$$

参数:
- $h$: 头数(LLaMA-7B: 32; DeepSeek-V3: 128)
- $W_Q^i, W_K^i \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_V^i \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W_O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

### 7.6.2 实现优化:合并投影

朴素实现需要 $h$ 次独立投影。优化做法:**用一个大矩阵一次性投影**:

$$
\tilde{Q} = X W_Q^{\text{all}}, \quad W_Q^{\text{all}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

然后将 $\tilde{Q}$ reshape 成 $[n, h, d_k]$:

```python
Q = (X @ W_Q_all).view(n, h, d_k).transpose(1, 2)   # [h, n, d_k]
```

K、V 同理。每个头独立做 attention,最后再合并。

### 7.6.3 多头的几何含义

每个头可以理解为在**不同的子空间**中做 attention:

- 头 1 可能学习"主语-谓语"关系
- 头 2 可能学习"指代-被指代"关系
- 头 3 可能学习"距离衰减"模式
- ...

**关键洞察**: 不同头的功能是**训练自发涌现**的,没有人工指定。Anthropic 等机构的 interpretability 研究已识别出许多可解释的"专家头"(参见 §10 延伸阅读)。

### 7.6.4 多头数量怎么选?

经验法则:
- $h \cdot d_k = d_{\text{model}}$
- $d_k$ 通常 64 或 128(过小表达力不足,过大效率低)
- $h$ 通常 8~128,大模型更多

例:
- BERT-base: $h=12$, $d_k=64$
- GPT-3: $h=96$, $d_k=128$
- LLaMA-7B: $h=32$, $d_k=128$
- DeepSeek-V3: $h=128$ (实际使用 MLA,见 §24)

---

## 7.7 完整代码实现

下面是一个**可运行**的 Multi-Head Attention 实现(约 50 行):

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        
        # 合并投影:用一个大矩阵一次性算 Q/K/V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        return: [batch, seq_len, d_model]
        """
        B, T, _ = x.shape
        
        # 1. 投影 + reshape 多头  → [B, h, T, d_k]
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled dot-product
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)   # [B, h, T, T]
        
        # 3. Causal mask
        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # 4. Softmax + 加权
        attn = F.softmax(scores, dim=-1)
        out  = attn @ v                                            # [B, h, T, d_k]
        
        # 5. 合并头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


# === 测试 ===
if __name__ == "__main__":
    attn = MultiHeadAttention(d_model=512, n_heads=8, causal=True)
    x = torch.randn(2, 10, 512)
    y = attn(x)
    print(y.shape)  # torch.Size([2, 10, 512])
```

可以直接复制运行,验证输入输出形状。

---

## 7.8 复杂度分析

### 7.8.1 时间复杂度

| 操作 | 复杂度 |
|---|---|
| $QK^\top$ | $O(n^2 d)$ |
| Softmax | $O(n^2)$ |
| $A V$ | $O(n^2 d)$ |
| **总计** | $\boxed{O(n^2 d)}$ |

其中 $n$ 是序列长度,$d = d_{\text{model}}$。

**关键洞察**: Attention 对序列长度 $n$ 是**二次复杂度**。这是长上下文模型的最大瓶颈。

### 7.8.2 空间复杂度

注意力矩阵 $A \in \mathbb{R}^{n \times n}$ 占据 $O(n^2)$ 显存。对于:
- $n = 4096$, fp16: 32 MB
- $n = 32768$, fp16: **2 GB**
- $n = 128000$, fp16: **31 GB**

这就是为什么长上下文模型需要 Flash Attention(避免显式存储 $A$,参见 §21)和 MLA(压缩 KV cache,参见 §24)。

### 7.8.3 训练 vs 推理的差异

- **训练**: 一次性处理整个序列,attention 复杂度 $O(n^2 d)$
- **推理 Prefill**: 同训练
- **推理 Decode**: 每次只算 1 个 query,attention 复杂度 $O(n d)$,但需要存 $O(n d)$ 的 KV cache

详见 §18 推理流程, §19 KV Cache。

---

## 7.9 与其他变种的对比

后续章节会深入,这里仅作铺垫:

| 变种 | 改进点 | 详见 |
|---|---|---|
| **MHA (本章)** | baseline | §7 |
| **MQA** | 所有 head 共享 K, V | §24 引论 |
| **GQA** | head 分组,组内共享 K, V (LLaMA-2/3) | §24 引论 |
| **MLA** | KV 低秩压缩 (DeepSeek-V2/V3) | §24 主体 |
| **Sliding Window** | 局部 attention (Mistral) | §21 |
| **Flash Attention** | 不变语义,加速 IO | §21 |

---

## 🎨 7.9.A 通俗类比合集

抽象的公式背后,都是非常生活化的场景。用三组类比帮你把整章串起来。

### 类比一: 班级讨论会 🏫

> 一个班 20 个同学,老师让大家就一个话题各抒己见。

- **每个同学 = 一个 token**
- **同学心里的"问题"** = Query (Q): "关于这个话题,我想了解什么?"
- **同学的"标签"** = Key (K): "我擅长什么、关注什么"
- **同学的"发言"** = Value (V): "我的实际观点"

讨论过程:

1. 每个人把自己的"问题"(Q)广播给所有人
2. 比对自己的"问题"和别人的"标签"(K),决定**对谁更感兴趣**
3. 按兴趣度加权吸收每个人的"发言"(V)
4. 综合后形成自己的新理解

**Multi-Head = 同时开 8 个分组讨论会**: 一组讨论"主谓宾",一组讨论"情感色彩",一组讨论"时间顺序"…最后合并所有组的收获。

**Causal Mask = 老师规定**"只能引用前面同学的发言,后面还没说话的不能引用",防止剧透。

### 类比二: 求职 + 招聘 💼

把 Attention 看作一场双向选择:

- **求职者(每个 token)的简历** = Q
- **公司的 JD(职位描述)** = K
- **公司能给的实际资源/培训/项目** = V

匹配流程:

1. 简历 Q 和每家公司的 JD (K) 算"匹配度" (`QK^T`)
2. 除以 √d_k = **按行业平均水平做归一**,防止某个高薪岗让所有人挤过去
3. Softmax = **匹配度变成"投递概率"**(加起来必须 100%)
4. 按概率加权拿到各公司的资源 (V) = "决定从每家学到多少"

**Multi-Head = 求职者同时考虑技术、薪资、地理、文化等多维度**,每维独立打分,综合决策。

### 类比三: 图书馆查资料 📚

写论文时从 1000 本书里提取信息:

| 步骤 | 对应公式 |
|---|---|
| 心里默念关键词 | Q |
| 看每本书脊的标签 | K |
| 拿到书的实际内容 | V |
| 关键词 × 标签 = 相关分数 | QK^T |
| 标签数量多,分数大,要归一化 | / √d_k |
| 把分数转成"读多大比例" | softmax |
| 按比例读各书,综合写论文 | A · V |

**Causal Mask 的图书馆版本**: 书按出版时间排序,写 2020 年的论文不能引用 2024 年的书。

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
|---|---|---|
| 🏫 班级讨论 | **token 间互相交互** | Self-Attention 的"自"在哪 |
| 💼 求职招聘 | **匹配与归一** | √d_k 与 softmax 的作用 |
| 📚 图书馆 | **加权信息融合** | Q/K/V 各自的角色 |

**记忆口诀**:
> **"提问找标签,匹配定权重,加权融内容。"**

---

## 7.10 常见疑问

### Q1: Q 和 K 数学上对称,为什么要分两个矩阵?
**A**: $W_Q$ 和 $W_K$ 学习的是**两种不同的角色**——前者学"如何提问",后者学"如何被检索"。即使数学形式对称,语义角色不对称,需要独立参数。

### Q2: 为什么 V 也要投影,不能直接用 $X$?
**A**: $V$ 控制"提供什么内容"。直接用 $X$ 等于"提供原始信息",限制了模型表达。$W_V$ 让模型可以学到"提供经过加工后的特征"。

### Q3: Softmax 之后再 mask 行不行?
**A**: 不行。Softmax 是归一化操作,如果先做 softmax 再 mask 掉部分位置置零,剩余权重和不为 1。**正确做法是先 mask 再 softmax**。

### Q4: Attention 矩阵不是对称的吗?
**A**: 不是。$QK^\top$ 一般不对称,因为 $Q \neq K$(它们由不同投影矩阵生成)。即使在 self-attention 中也不对称。

### Q5: 多头之间会冗余吗?
**A**: 有可能。研究表明许多头是冗余的,可以剪枝(参见 *"Are Sixteen Heads Really Better than One?"*, Michel et al., 2019)。这也是 MQA/GQA/MLA 的动机之一。

---

## 7.11 本章小结

### 核心公式
**Scaled Dot-Product Attention**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**Multi-Head Attention**:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

### 核心要点

| 要点 | 关键 |
|---|---|
| **本质** | 基于相似度的加权求和 |
| **Q/K/V** | 来自同一输入的三种投影 |
| **缩放 √d_k** | 防止 softmax 饱和,数值稳定 |
| **Causal Mask** | 实现自回归生成 |
| **Multi-Head** | 并行学习多种关系模式 |
| **复杂度** | $O(n^2 d)$,长序列瓶颈 |

### 一句话总结

> **Self-Attention = 每个 token 用自己的 Query 对所有 token 的 Key 打分,Softmax 归一为权重,再对所有 Value 加权求和。这一机制让任意距离的 token 一步直达,是 Transformer 区别于 RNN 的核心创新。Multi-Head 则让模型并行学习多种关系模式,Causal Mask 保证生成的因果性,$\sqrt{d_k}$ 缩放保证训练稳定。**

---

## 7.12 延伸阅读

### 必读论文
1. Vaswani et al., 2017. *"Attention is All You Need"*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Michel et al., 2019. *"Are Sixteen Heads Really Better than One?"*. NeurIPS. [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)

### 进阶论文
3. Shazeer, 2019. *"Fast Transformer Decoding: One Write-Head is All You Need"*. (MQA 起源)
4. Ainslie et al., 2023. *"GQA: Training Generalized Multi-Query Transformer Models"*.
5. DeepSeek-AI, 2024. *"DeepSeek-V2"*. (MLA 提出)

### 推荐资源
- 3Blue1Brown YouTube: *"But what is GPT?"* — 极佳的可视化解释
- Lilian Weng: *"The Transformer Family"* 博客系列
- Karpathy: *"Let's build GPT: from scratch"* — 从零实现

---

## 7.13 实战练习

### 练习 1 (★)
修改 §7.7 的代码,把 causal mask 改成 bidirectional(无 mask),观察 BERT 风格的 attention 行为。

### 练习 2 (★★)
实现一个"局部 attention" —— 每个 token 只看前 $w$ 个 token(滑动窗口)。

### 练习 3 (★★★)
分析 $QK^\top$ 矩阵的几何含义:在什么条件下,attention 会退化为"邻居平均"(类似 CNN)?在什么条件下会退化为"全局平均"?

### 练习 4 (★★★)
推导 attention 梯度:$\frac{\partial \mathcal{L}}{\partial Q}$、$\frac{\partial \mathcal{L}}{\partial K}$、$\frac{\partial \mathcal{L}}{\partial V}$ 的形式。

---

## 章节交叉引用

- 前置: [§01 线性代数](../Part-I_数学与基础/01_线性代数与矩阵运算.md), [§06 Embedding 与位置编码](06_Embedding与位置编码.md)
- 后续: [§08 FFN 与激活函数](08_FFN与激活函数.md), [§19 KV Cache](../Part-IV_推理与部署/19_KV-Cache与显存管理.md), [§24 MLA 深度解析](../Part-V_DeepSeek专题/24_MLA深度解析.md)
- 相关: [§09 归一化与残差](09_归一化与残差.md), [§21 推理优化(Flash Attention)](../Part-IV_推理与部署/21_推理优化.md)

---

*下一章: §08 FFN 与激活函数*
