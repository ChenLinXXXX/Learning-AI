# 06 Embedding 与位置编码(Embedding / 绝对 / 正弦 / RoPE / ALiBi / YaRN)

> **章节定位**: Part II 入门→核心的桥梁,把 token ID 变成可计算的向量并注入"位置"信息
> **预计阅读时间**: 60-90 分钟
> **难度**: ★★★☆☆(RoPE/YaRN 部分 ★★★★)
> **前置知识**: §01 线性代数(矩阵/向量,旋转矩阵)、§05 Tokenization
> **后续依赖**: §07 Self-Attention、§10 Block 整体、§19 KV Cache、§24 MLA

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [6.1 Embedding:从整数到向量](#61-embedding从整数到向量)
- [6.2 为什么需要位置编码?](#62-为什么需要位置编码)
- [6.3 绝对位置编码:学习式 vs 正弦](#63-绝对位置编码学习式-vs-正弦)
- [6.4 相对位置编码(简介)](#64-相对位置编码简介)
- [6.5 RoPE:旋转位置编码](#65-rope旋转位置编码)
- [6.6 ALiBi:用线性偏置代替位置编码](#66-alibi用线性偏置代替位置编码)
- [6.7 长度外推问题与 YaRN](#67-长度外推问题与-yarn)
- [6.8 完整代码实现](#68-完整代码实现)
- [6.9 工程对比与选型](#69-工程对比与选型)
- [🎨 6.9.A 通俗类比合集](#-69a-通俗类比合集)
- [6.10 常见疑问](#610-常见疑问)
- [6.11 本章小结](#611-本章小结)
- [6.12 延伸阅读](#612-延伸阅读)
- [6.13 实战练习](#613-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

Tokenizer 输出的是整数序列,但 Transformer 内部全是连续向量运算。本章解决两件事:

1. **Embedding**: 把 token ID 映射成稠密向量
2. **位置编码**: 让模型知道"谁先谁后"——因为 Self-Attention 本身是**置换不变**的

我们将从最简单的 sinusoidal 出发,经过相对位置编码,推导出现代 LLM 普遍采用的 **RoPE(旋转位置编码)**,讨论 **ALiBi** 的另一种思路,最后落到长上下文工程问题与 **YaRN** 的频率缩放方案。

---

## 学习目标

完成本章后,你能够:

- ✅ 解释 Embedding 与 LM head 通常**共享权重**的原因 (§6.1.3)
- ✅ 推导为什么 Self-Attention 没有位置信息时是置换不变 (§6.2)
- ✅ 写出 sinusoidal 位置编码完整公式 (公式 6.3)
- ✅ 写出 RoPE 的旋转矩阵与点积性质 (公式 6.7–6.8)
- ✅ 解释 ALiBi 为什么天然有外推能力 (§6.6)
- ✅ 解释 YaRN 与"NTK / 线性插值"在频率域的关系 (§6.7)
- ✅ 用 PyTorch 写一个最小 RoPE 实现 (§6.8.2)

---

## 6.1 Embedding:从整数到向量

### 6.1.1 定义

给定词表 $V$ 与隐藏维度 $d_{\text{model}}$,Embedding 矩阵:

$$
E \in \mathbb{R}^{V \times d_{\text{model}}} \tag{6.1}
$$

对一个 token id $i$,查表得向量 $E_i \in \mathbb{R}^{d_{\text{model}}}$。

工程上等价于一个无 bias 的 `nn.Linear(V, d_model)`,但因为输入是 one-hot,直接索引比矩阵乘快得多——这就是 `nn.Embedding` 的实现。

### 6.1.2 初始化

常见做法:

- $E_{ij} \sim \mathcal{N}(0, 0.02)$(GPT/BERT 默认)
- LLaMA / DeepSeek 用 truncated normal,std 与 $d_{\text{model}}$ 相关

### 6.1.3 与 LM head 的"权重绑定"

输出端的 LM head 把 $h \in \mathbb{R}^{d_{\text{model}}}$ 映射回词表 logits:

$$
\text{logits} = h W_{\text{lm}}^\top, \quad W_{\text{lm}} \in \mathbb{R}^{V \times d_{\text{model}}} \tag{6.2}
$$

**权重绑定(tie weights)** 把 $W_{\text{lm}} = E$,**节省 $V \cdot d_{\text{model}}$ 参数**(对 GPT-2 来说约 38M)。

- BERT、GPT-2、T5:✅ 绑定
- LLaMA-1:✅ 绑定;LLaMA-2/3、DeepSeek-V3:❌ 解绑(独立 LM head,更灵活,大模型代价可忽略)

### 6.1.4 输入侧的归一化技巧

GPT 系列在 embedding 之后乘 $\sqrt{d_{\text{model}}}$(来自原论文),让 embedding 与位置编码量级匹配。现代 LLM(Pre-RMSNorm 派)通常**不做**这一步,因为第一层的 RMSNorm 会自动归一(§09)。

---

## 6.2 为什么需要位置编码?

### 6.2.1 Self-Attention 的"置换不变性"

Attention 公式:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

如果同时打乱 $Q, K, V$ 的行顺序(置换矩阵 $P$),输出**只是相应行的置换**,**值不变**。

换言之:如果不告诉模型"位置",`"我打你"` 和 `"你打我"` 的 token 集合相同,Attention 给出的内部表示也一样。这显然不对。

### 6.2.2 解决方案三大派

| 派系 | 思路 | 代表 |
| --- | --- | --- |
| 加法绝对位置 | 给 embedding 加上 $p_i$ | Transformer 原版 / BERT / GPT-2 |
| 修改 attention 偏置 | 在 $QK^\top$ 里加距离项 | ALiBi、T5 relative bias |
| 旋转 Q/K | 用复数旋转使内积带位置信息 | **RoPE**(LLaMA、Qwen、DeepSeek、Mistral…) |

现代 LLM 几乎一致采用 RoPE(理由见 §6.5.4)。

---

## 6.3 绝对位置编码:学习式 vs 正弦

### 6.3.1 学习式(BERT 风格)

直接学一张位置 embedding 表 $P \in \mathbb{R}^{n_{\max} \times d}$:

$$
\text{input}_i = E_{x_i} + P_i
$$

**缺点**:无法外推到 $n > n_{\max}$ 的序列,长度被硬编码。

### 6.3.2 正弦位置编码(原版 Transformer)

Vaswani et al., 2017:

$$
\begin{aligned}
P_{i, 2k} &= \sin\!\big(i / 10000^{2k/d}\big) \\
P_{i, 2k+1} &= \cos\!\big(i / 10000^{2k/d}\big)
\end{aligned} \tag{6.3}
$$

每个维度对应一个不同频率的正弦/余弦,频率从 1 到 $1/10000$ 指数变化。

**性质**:

- 形式上是**绝对位置**,但 $\sin / \cos$ 的加法定理让 $P_{i+\delta}$ 可以表示成 $P_i$ 的线性组合 → 隐含相对位置信息
- 不需要学习,任意 $i$ 都能算出来 → **理论上可外推**

**实践问题**:外推能力其实不强,$i > $ 训练长度时模型精度断崖式下降。

---

## 6.4 相对位置编码(简介)

直接把"位置之间的距离"作为 attention 的输入。代表方法:

### 6.4.1 Shaw / Transformer-XL

把 $i - j$ 通过一个可学习的偏置加到 $QK^\top$ 上:

$$
A_{ij} = \frac{Q_i K_j^\top + b_{i-j}}{\sqrt{d_k}} \tag{6.4}
$$

### 6.4.2 T5 风格

把相对距离桶化(0, 1, 2, 3-4, 5-8, 9-16…),每桶学一个标量加到 attention 上。

**优点**:外推更稳;**缺点**:对 KV cache 不友好(每对位置都要查表),工程性差。RoPE 和 ALiBi 之所以胜出,是因为它们既保留相对位置语义,又对 KV cache 极友好。

---

## 6.5 RoPE:旋转位置编码

### 6.5.1 核心思想

Su et al., 2021。**不加偏置,也不加 embedding,而是把位置编码"乘"进 Q 和 K**——通过一个**与位置相关的旋转矩阵**。

把 $d_k$ 维向量两两配对成 $d_k/2$ 个 2D 复数,每对 $(x_{2k}, x_{2k+1})$ 看成一个复平面上的点;给位置 $i$ 的向量旋转角度 $i \theta_k$:

$$
\theta_k = 10000^{-2k/d_k}, \quad k = 0, 1, \ldots, d_k/2 - 1 \tag{6.5}
$$

旋转矩阵:

$$
R_i^{(k)} = \begin{pmatrix} \cos(i\theta_k) & -\sin(i\theta_k) \\ \sin(i\theta_k) & \phantom{-}\cos(i\theta_k) \end{pmatrix} \tag{6.6}
$$

把 Q、K 的每一对维度按 $R_i$ 旋转,记作 $\tilde Q_i = R_i Q_i$、$\tilde K_j = R_j K_j$。

### 6.5.2 关键性质:内积只依赖相对位置

对旋转后向量做内积:

$$
\langle \tilde Q_i, \tilde K_j \rangle = \langle R_i Q_i, R_j K_j \rangle = Q_i^\top R_{j-i} K_j \tag{6.7}
$$

— 完全等价于"把 K 旋转 $j-i$ 再与 Q 做点积"。**绝对旋转产生相对距离**,这是 RoPE 的优雅之处。

**含义**:Attention 分数自然只对相对距离 $j - i$ 敏感,而工程上只需要对 Q、K 分别作用一次,**无需修改 attention 主公式,也无需新增参数**。

### 6.5.3 工程实现的小窍门

代码里更常用的"分半"实现(LLaMA 派):把 $d_k$ 维向量按下标分成前一半 / 后一半,而不是奇偶配对,等价但 reshape 更简单:

```text
x = [x_0, x_1, ..., x_{d-1}]
rotate_half(x) = [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]
RoPE(x, i) = x * cos(iθ) + rotate_half(x) * sin(iθ)
```

详见 §6.8.2 代码。

### 6.5.4 为什么 RoPE 赢了?

- ✅ 相对位置语义但无额外参数
- ✅ KV cache 友好:旋转只作用在 Q、K 上,V 不变,缓存逻辑不复杂
- ✅ 可外推(配合 NTK / YaRN,见 §6.7)
- ✅ 数学性质清晰,可被 Flash Attention 等加速

LLaMA、Mistral、Qwen、DeepSeek、Yi、ChatGLM 等几乎所有主流开源 LLM 都采用 RoPE。

---

## 6.6 ALiBi:用线性偏置代替位置编码

### 6.6.1 思路

Press et al., 2022。**完全去掉显式位置编码**,只在 attention 分数上加一个与距离成正比的负偏置:

$$
A_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_k}} - m \cdot |i - j| \tag{6.8}
$$

每个 head 用一个不同的斜率 $m$(几何序列,从 $2^{-1/h}$ 到 $2^{-h/h}$)。

### 6.6.2 直觉

距离越远,attention 越受惩罚——天然实现"局部偏好",并且**因为是固定斜率,完全不受训练长度限制 → 外推非常稳**。

### 6.6.3 现状

ALiBi 在 BLOOM、MPT 等模型上使用,工程极简洁。但在大规模 LLM 中,RoPE 因表达力更强、与 KV cache / Flash Attention 配合更顺,仍是事实标准。ALiBi 更多见于"追求快速长上下文外推"的场景。

---

## 6.7 长度外推问题与 YaRN

### 6.7.1 现象

RoPE 训练在 4k 上下文,推理时直接喂 32k → 模型崩溃(困惑度爆炸、输出乱码)。

原因:高频维度(小 $\theta_k$ 对应慢转一圈)在长序列上还好,但**低频维度**(大 $\theta_k$ 对应快转)在长 $i$ 处会落入训练时**从未见过的相位**,模型不知所措。

### 6.7.2 三种缓解方案

| 方法 | 思路 |
| --- | --- |
| **位置线性插值 (PI)** | 把位置 $i$ 缩成 $i / s$,$s$ 是放大倍数;等效压缩 RoPE 频率 |
| **NTK-aware 缩放** | 不缩位置,而是把 $10000$ 这个 base 改大,**只压低频维度**,保留高频细节 |
| **YaRN** | NTK 的精细化版本,**对不同频段做不同缩放**并加温度补偿 |

### 6.7.3 YaRN 关键公式(简化版)

YaRN(Peng et al., 2023)对 RoPE 频率分段处理:

- 高频($\theta_k$ 大):**不缩放**(细节保留)
- 低频($\theta_k$ 小):**线性插值压缩**(避免出现训练时未见过的相位)
- 过渡区:平滑插值

并在 attention 上加一个温度因子 $t$ 补偿 softmax 的尺度变化:

$$
A = \text{softmax}\!\left(\frac{1}{t \sqrt{d_k}} \tilde Q \tilde K^\top\right), \quad t = 0.1 \ln s + 1
$$

效果:把 LLaMA-7B 从 4k 外推到 128k,困惑度仍接近原训练长度。DeepSeek-V3 把上下文从 4k 扩到 128k 也用了 YaRN。

### 6.7.4 何时该用 YaRN?

- **训练时**:模型设计上想低成本支持长上下文,先小窗口预训练 + YaRN 续训
- **推理时**:仅改 RoPE 的频率参数,不改权重,**zero-shot 扩展到几十倍上下文**(质量略降但可用)

---

## 6.8 完整代码实现

### 6.8.1 Sinusoidal 位置编码

```python
import math
import torch

def sinusoidal_pe(n: int, d: int) -> torch.Tensor:
    """返回 [n, d] 的正弦位置编码。"""
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()        # [n, 1]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
```

### 6.8.2 RoPE 最小实现(LLaMA 风格)

```python
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        # θ_k = base^{-2k/dim}
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()             # 位置 0..max_len-1
        freqs = torch.outer(t, inv_freq)              # [max_len, dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        # 拼成 [max_len, dim]: 前后半各重复一份
        self.register_buffer("cos", torch.cat([cos, cos], dim=-1))
        self.register_buffer("sin", torch.cat([sin, sin], dim=-1))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim] → 前半与后半互换并对前半取负
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """
        q, k: [B, h, T, dim]
        offset: 起始位置(prefix/decoding 用)
        """
        T = q.size(-2)
        cos = self.cos[offset:offset + T]            # [T, dim]
        sin = self.sin[offset:offset + T]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# === 测试 ===
if __name__ == "__main__":
    rope = RoPE(dim=64, max_len=128)
    q = torch.randn(1, 4, 16, 64)
    k = torch.randn(1, 4, 16, 64)
    qr, kr = rope(q, k)
    print(qr.shape, kr.shape)   # torch.Size([1, 4, 16, 64])
```

### 6.8.3 ALiBi 偏置

```python
def alibi_bias(n: int, n_heads: int) -> torch.Tensor:
    """返回 [n_heads, n, n] 的 ALiBi 偏置矩阵。"""
    slopes = 2 ** -torch.arange(1, n_heads + 1) * 8 / n_heads
    pos = torch.arange(n)
    dist = (pos[None, :] - pos[:, None]).abs().float()   # [n, n]
    return -slopes[:, None, None] * dist                 # [h, n, n]
```

---

## 6.9 工程对比与选型

| 方案 | 加在哪 | 参数 | 外推 | KV cache 友好 | 代表 |
| --- | --- | --- | --- | --- | --- |
| 学习式绝对 | embedding | $n_{\max} \cdot d$ | ❌ 差 | ✅ | BERT |
| Sinusoidal | embedding | 无 | 一般 | ✅ | 原版 Transformer |
| T5 相对偏置 | attention | $O(\log n)$ 桶 | 一般 | 一般 | T5 |
| **RoPE** | Q, K | 无 | 中(+YaRN 极强) | ✅✅ | LLaMA / DeepSeek / Mistral |
| **ALiBi** | attention 分数 | 无 | ✅ 强 | ✅ | BLOOM, MPT |

**结论**:新模型设计,几乎默认 **RoPE + YaRN**;追求极简和稳健外推时考虑 ALiBi。

---

## 🎨 6.9.A 通俗类比合集

### 类比一: 教室座位号 + 学生头像 🪑

> Embedding 是"学生头像",位置编码是"座位号"。

- **Embedding**:每个学生(token)都有一张专属头像
- **绝对位置编码**:在头像下贴一张"座位号"标签 → 加法
- **RoPE**:**给头像戴一个随时间转动的指针**(像钟表),不同座位指针指向不同方向 → 乘法/旋转
- **ALiBi**:不贴标签也不戴指针,但**远距离同学的发言一律打折扣** → attention 偏置

### 类比二: 钟表盘上的指针 🕒

> RoPE 让每个维度对就像一个不同表盘:秒针、分针、时针、日历…

- $\theta_k$ **大** = 秒针,转得快,捕捉**短距离**精细差别
- $\theta_k$ **小** = 时针、日历,转得慢,捕捉**长距离**粗粒度
- 内积"对表"时,只看**两块表相差的角度**,不在乎绝对时刻 → 相对位置

YaRN = **加大日历类(慢针)的表盘**,让它在长上下文也不会绕回起点。

### 类比三: 排队取号叫号系统 🎫

> 银行叫号:每个客户拿一个号(位置),但服务员关心的其实是"在你前面还有几位"。

- **绝对位置编码**:每人手上写着 "我是第 1234 号"
- **相对位置(RoPE/ALiBi)**:服务员心里算"你和我目前在叫的号差几号" → 距离比绝对编号更稳健
- **ALiBi 的惩罚**:"号差越大,我对你越不耐烦" — 自然的衰减
- **RoPE 的旋转**:用两个旋转的指针互相对齐角度计算"号差",数学优雅

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🪑 教室座位 | **加法 vs 乘法** | 绝对 vs 旋转的根本差别 |
| 🕒 钟表 | **多频段** | RoPE 的"分维度旋转" |
| 🎫 排队 | **相对位置** | 为什么相对位置更鲁棒 |

**记忆口诀**:
> **"Embedding 给身份,RoPE 给指针,内积只读角差。"**

---

## 6.10 常见疑问

### Q1: 既然 Self-Attention 是置换不变,FFN 不也是位置无关吗?为什么 FFN 不用位置编码?
**A**: FFN 是 position-wise,每个位置独立处理但**共享同一组参数**,本来就不区分位置。位置信息必须从 attention 输入里携带过来——这就是为什么位置编码加在 Q、K 上(或 embedding 上),而不是加在 FFN 输入上。

### Q2: RoPE 加在 Q、K 上,V 为什么不加?
**A**: 关键性质在内积 $\tilde Q^\top \tilde K = Q^\top R_{j-i} K$,只需要 Q、K 旋转就能让 attention 分数依赖相对距离。V 是"信息内容",对它做旋转反而会破坏内容语义。

### Q3: RoPE 的 base = 10000 是怎么定的?改了会怎样?
**A**: 原论文沿用 sinusoidal 的 10000。base 越大,**频率谱越宽,可表达的长距离越长**——这正是 NTK-aware 外推的核心(把 base 改到 50万),YaRN 是其精细化版本。

### Q4: 不同 head 要不要用不同的位置编码?
**A**: 一般所有 head 共用一套 RoPE(每 head 独立做旋转就好);ALiBi 则相反——**每 head 用不同斜率**才能学到不同尺度的偏好。

### Q5: 训练时改 base / 改 max_len 后,推理时 KV cache 怎么办?
**A**: KV cache 里存的是**已旋转过的 K**(LLaMA 实现)或**未旋转的 K + 当前 offset**(部分实现)。改 RoPE 频率参数 = 改变了旋转角度,缓存中已存的 K 就**作废**,必须重新计算;否则会引入静默错误。生产代码要严格管理这点。

---

## 6.11 本章小结

### 核心公式

**Sinusoidal**:

$$
P_{i,2k} = \sin(i / 10000^{2k/d}), \quad P_{i,2k+1} = \cos(i / 10000^{2k/d})
$$

**RoPE 内积性质**:

$$
\langle R_i q, R_j k \rangle = q^\top R_{j-i} k
$$

**ALiBi**:

$$
A_{ij} \mathrel{+}= -m_h \cdot |i - j|
$$

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **Embedding** | $V \times d$ 查找表,常与 LM head 共享 |
| **置换不变性** | 必须显式注入位置 |
| **绝对 vs 相对** | 相对更鲁棒、对 KV cache 友好 |
| **RoPE** | 通过旋转把绝对位置变为相对内积 |
| **YaRN** | 频率分段缩放,实现长上下文外推 |
| **ALiBi** | attention 分数加负距离偏置,工程极简 |

### 一句话总结

> **Embedding 把 token ID 翻译成稠密向量;Self-Attention 本身不知道位置,所以必须显式注入。现代 LLM 的事实标准是 RoPE——把绝对位置编码成 Q、K 上的旋转,内积自然变成只对相对距离敏感;长上下文外推则通过 YaRN 在频率域分段缩放实现。**

---

## 6.12 延伸阅读

### 必读论文

1. Vaswani et al., 2017. *"Attention is All You Need"*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)(Sinusoidal)
2. Su et al., 2021. *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)(RoPE 原论文)
3. Press et al., 2022. *"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"*. ICLR. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)(ALiBi)

### 进阶论文

4. Chen et al., 2023. *"Extending Context Window of LLMs via Position Interpolation"*. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)(线性插值)
5. bloc97, 2023. *"NTK-Aware Scaled RoPE"*(Reddit / 博客)
6. Peng et al., 2023. *"YaRN: Efficient Context Window Extension of Large Language Models"*. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
7. Shaw et al., 2018. *"Self-Attention with Relative Position Representations"*. NAACL.

### 推荐资源

- EleutherAI 博客: *"Rotary Embeddings: A Comprehensive Reference"*
- HuggingFace: *"Long Context: A Survey of Position Encodings"*

---

## 6.13 实战练习

### 练习 1 (★)

可视化 sinusoidal 位置编码:画一张 $n=128$、$d=64$ 的热力图,观察不同维度的周期差异。

### 练习 2 (★)

修改 §6.8.2 的 `RoPE`,把 `base` 改为 50万,与默认 10000 在 8192 长度上的相位差异作图。

### 练习 3 (★★)

将 §6.8.2 的 RoPE 接入 §07 的 `MultiHeadAttention`,在合成数据(token 1..100 的累加任务)上对比"无位置编码 / sinusoidal / RoPE"三种方案的训练 loss。

### 练习 4 (★★★)

实现一个简化版 YaRN:对 RoPE 的高频部分保留,对低频部分按 PI 缩放,并用一个温度 $t = 0.1\ln s + 1$ 补偿 attention。在 LLaMA-2-7B 上做 zero-shot 32k 推理(只改 RoPE,不改权重),记录困惑度变化。

### 练习 5 (★★★)

推导 ALiBi 与 RoPE 在数学上是否能等价:能否找到一个 RoPE 频率配置使得 attention 分数衰减形态接近 ALiBi?给出理论分析与数值验证。

---

## 章节交叉引用

- 前置: [§05 Tokenization](05_Tokenization.md), [§01 线性代数](../Part_I_数学与基础/01_线性代数与矩阵运算.md)
- 后续: [§07 Self-Attention](07_Self_Attention机制.md), [§10 Transformer Block 整体](10_Transformer_Block整体.md)
- 相关: [§19 KV Cache](../Part_IV_推理与部署/19_KV_Cache与显存管理.md)(RoPE 在缓存中的存法), [§24 MLA](../Part_V_DeepSeek专题/24_MLA深度解析.md)(MLA 中的 decoupled RoPE)

---

*下一章: §07 Self-Attention 机制(详见示范章节)*
