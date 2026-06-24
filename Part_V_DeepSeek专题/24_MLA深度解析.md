# 24 MLA 深度解析(Multi-head Latent Attention:低秩 KV + Decoupled RoPE)

> **章节定位**: Part V 第 2 章,DeepSeek 架构创新的"压舱石"
> **预计阅读时间**: 80-110 分钟
> **难度**: ★★★★☆
> **前置知识**: §07 Self-Attention、§01 SVD/低秩、§06 RoPE、§19 KV cache
> **后续依赖**: §26 V3 工程、§27 V3 vs LLaMA-3

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [24.1 动机:KV cache 是长上下文的拦路虎](#241-动机kv-cache-是长上下文的拦路虎)
- [24.2 思路一:朴素低秩 KV(失败之路)](#242-思路一朴素低秩-kv失败之路)
- [24.3 MLA 完整公式](#243-mla-完整公式)
- [24.4 RoPE 的兼容性问题与 Decoupled RoPE](#244-rope-的兼容性问题与-decoupled-rope)
- [24.5 推理时的"矩阵吸收"](#245-推理时的矩阵吸收)
- [24.6 KV cache 大小与压缩比](#246-kv-cache-大小与压缩比)
- [24.7 复杂度分析](#247-复杂度分析)
- [24.8 完整代码实现](#248-完整代码实现)
- [24.9 与 MHA / MQA / GQA 的对照](#249-与-mha--mqa--gqa-的对照)
- [24.10 V3 中的实际配置](#2410-v3-中的实际配置)
- [🎨 24.10.A 通俗类比合集](#-2410a-通俗类比合集)
- [24.11 常见疑问](#2411-常见疑问)
- [24.12 本章小结](#2412-本章小结)
- [24.13 延伸阅读](#2413-延伸阅读)
- [24.14 实战练习](#2414-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

**MLA(Multi-head Latent Attention)** 是 DeepSeek-V2/V3 的核心架构创新,把 KV cache 压成低秩潜变量,**长上下文显存降 1-2 个数量级**,且推理时通过"矩阵吸收"几乎不引入额外算力。本章按下面四步拆解:

1. **直觉**:KV 矩阵在大模型里普遍低秩,可压缩
2. **公式**:把 KV 投影到 $c_t \in \mathbb{R}^{d_c}$($d_c \ll d_{\text{model}}$),用两个上投影还原
3. **难点**:RoPE 与低秩压缩冲突 → **decoupled RoPE**(K 分成"语义部分 + 位置部分"双通道)
4. **推理优化**:矩阵吸收消去 explicit K/V,推理 attention 直接作用在 $c_t$ 上

读完你应能写出 MLA 的 forward / KV cache 布局,并解释为什么 V3 671B / 37B 激活的服务成本仍可控。

---

## 学习目标

完成本章后,你能够:

- ✅ 写出 MLA 的 K/V 压缩公式(公式 24.2-24.3)
- ✅ 解释"为什么 RoPE 不能直接作用在压缩后的 c 上" (§24.4.2)
- ✅ 推导矩阵吸收(W_K · W_Q^T 合并)的等价性 (§24.5.2)
- ✅ 计算 DeepSeek-V3 单 token KV cache 字节(§24.6 表)
- ✅ 写 50 行简化 MLA forward 代码(§24.8)

---

## 24.1 动机:KV cache 是长上下文的拦路虎

回顾 §19.2:对 MHA 模型,KV cache 大小与 $L \cdot n \cdot h_{\text{kv}} \cdot d_k$ 成正比。LLaMA-2-7B 在 128k 上下文时 KV cache 达 64 GB——**比权重还大**。

之前的缓解:

- **MQA**:压 KV 到 $h$ 倍小,但 head 间共享 K/V 损质量
- **GQA**:折中,LLaMA-3 用 8 组,KV 压 8 倍
- **Sliding Window**:丢历史,丢质量

DeepSeek 想要**不损质量、显著更小的 KV cache**,于是提出 MLA。

---

## 24.2 思路一:朴素低秩 KV(失败之路)

### 24.2.1 直觉

LLM 的 K, V 投影矩阵实证上低秩(§01.5.2)。如果 K, V 行向量本身就低秩,**只缓存"潜变量" $c_t$,需要时再展开**:

$$
c_t = W_{DKV}\, h_t \in \mathbb{R}^{d_c}, \quad d_c \ll d \tag{24.1}
$$

decode 时:用 $W_{UK}$, $W_{UV}$ 把 $c_t$ 还原成 $K_t$, $V_t$:

$$
K_t = W_{UK}\, c_t, \quad V_t = W_{UV}\, c_t
$$

### 24.2.2 看似可行,问题在哪?

如果 K 直接由 $c_t$ 还原,**RoPE 要旋转 $K_t$**,得 $K_t' = R_t \cdot W_{UK} c_t$。在推理时,$R_t$ 与位置相关 → **不能在训练时把 $W_{UK}$ 跟 $W_Q$ 提前合并**(因为 $R_t$ 中插一脚)→ "矩阵吸收"失效,每步都要先展开成完整 $K_t$,**KV cache 压不下**。

朴素低秩 K → 与 RoPE 冲突。需要更巧的设计。

---

## 24.3 MLA 完整公式

为解决上述冲突,DeepSeek 把 K 拆成**两部分**:

- **语义部分** $k^{(C)}_t$:由低秩潜变量 $c_t$ 上投影得到,**不加 RoPE**
- **位置部分** $k^{(R)}_t$:单独投影出来 + 加 RoPE,**所有 head 共享**

V 也由 $c_t$ 上投影。Q 也类似拆分。

### 24.3.1 完整方程

记 hidden 维 $d$,head 数 $n_h$,每 head 维 $d_h = d_h^{(C)} + d_h^{(R)}$。

**KV 压缩与还原**:

$$
c^{KV}_t = W^{DKV} h_t \in \mathbb{R}^{d_c}
$$

$$
[k^{(C)}_{t,1}; \ldots; k^{(C)}_{t,n_h}] = W^{UK}\, c^{KV}_t \in \mathbb{R}^{n_h \cdot d_h^{(C)}}
$$

$$
[v_{t,1}; \ldots; v_{t,n_h}] = W^{UV}\, c^{KV}_t \in \mathbb{R}^{n_h \cdot d_h^{(C)}}
$$

**K 的位置部分(所有 head 共享)**:

$$
k^{(R)}_t = \text{RoPE}\!\big(W^{KR}\, h_t\big) \in \mathbb{R}^{d_h^{(R)}} \tag{24.2}
$$

每 head 的 K 是拼接:

$$
k_{t,i} = \begin{bmatrix} k^{(C)}_{t,i} \\ k^{(R)}_t \end{bmatrix} \in \mathbb{R}^{d_h} \tag{24.3}
$$

**Q 端类似**(可选 Q 也做低秩,V2 中是;V3 中 Q 端没压缩,直接对 $h_t$ 投影):

$$
q_{t,i} = \begin{bmatrix} q^{(C)}_{t,i} \\ q^{(R)}_{t,i} \end{bmatrix}, \quad
q^{(C)}_{t,i} = W^{Q,C}_i h_t, \quad
q^{(R)}_{t,i} = \text{RoPE}\!\big(W^{Q,R}_i h_t\big)
$$

注意 Q 的位置部分**每 head 各算一份**,K 的位置部分**所有 head 共享**——这是不对称设计,**显著节省 KV cache 但不损质量**。

**Attention 计算与 MHA 一致**:

$$
\text{Attn}_i = \text{softmax}\!\Big(\frac{q_{t,i}^\top \cdot k_{:,i}}{\sqrt{d_h}}\Big) \cdot v_{:,i}
$$

### 24.3.2 KV cache 内容

每 token 需要缓存:

- $c^{KV}_t \in \mathbb{R}^{d_c}$(单份,所有 head 共享)
- $k^{(R)}_t \in \mathbb{R}^{d_h^{(R)}}$(单份,所有 head 共享)

**单 token KV 字节** = $(d_c + d_h^{(R)}) \cdot \text{bytes}$,与 $n_h$、$L$ 解耦(对每层独立)。

对照 MHA 单 token 单层:$2 \cdot n_h \cdot d_k \cdot \text{bytes}$ → 压缩比

$$
\rho = \frac{2 n_h d_k}{d_c + d_h^{(R)}} \tag{24.4}
$$

V3 默认 $n_h = 128$, $d_k = 128$, $d_c = 512$, $d_h^{(R)} = 64$:

$$
\rho = \frac{2 \cdot 128 \cdot 128}{512 + 64} = \frac{32768}{576} \approx 57\times
$$

即 KV cache 比同规模 MHA 压**~57 倍**(具体数字按 V3 实际配置略有差异,文献给出 ~10× 相对 GQA,~70× 相对 MHA)。

---

## 24.4 RoPE 的兼容性问题与 Decoupled RoPE

### 24.4.1 为什么不能把 RoPE 加到 $k^{(C)}$ 上?

如果 $k_t = \text{RoPE}_t(W^{UK} c_t)$,那 attention 算:

$$
q^\top k_t = (W^Q h)^\top R_t W^{UK} c_t
$$

$R_t$ 与位置相关,**无法把 $W^Q$ 与 $W^{UK}$ 静态合并**。推理时每步必须显式还原 $k_t \in \mathbb{R}^{d_h}$,KV cache 就回到 MHA 大小,压缩失效。

### 24.4.2 解耦的妙处

让 $k^{(C)}$ **不带 RoPE**,纯靠 $W^{UK}$ 把 $c_t$ 展开 → $W^Q W^{UK}$ 可以**离线合并成新矩阵 $W^{Q'} = W^{UK \top} W^Q$**(矩阵吸收,§24.5),attention 不再显式构造 $k^{(C)}_t$。

位置信息全部丢到 $k^{(R)}$ 上,这部分**仍是普通 RoPE**,但维度只有 $d_h^{(R)} = 64$,缓存代价低。Q 端对应的 $q^{(R)}$ 与 $k^{(R)}$ 做"位置内积"。最终 attention 分数:

$$
\text{score}_{t,s,i} = \underbrace{q^{(C)\,\top}_{t,i} k^{(C)}_{s,i}}_{\text{语义}} + \underbrace{q^{(R)\,\top}_{t,i} k^{(R)}_s}_{\text{位置}}
$$

两个内积各负其责。**等价于把 attention 拆成两个低维空间**:语义空间用低秩压缩,位置空间用 RoPE 走小维度。

### 24.4.3 为什么 $k^{(R)}$ 所有 head 共享?

如果每 head 各一份 $k^{(R)}_t$,KV cache 会多一项 $n_h \cdot d_h^{(R)}$ 字节,**重新爆炸**。共享是关键设计——实证表明位置信号在 head 间冗余度高,共享几乎无损。

---

## 24.5 推理时的"矩阵吸收"

### 24.5.1 朴素 forward

每 step decode:

1. 从 cache 读 $c^{KV}_s$ 与 $k^{(R)}_s$
2. 算 $k^{(C)}_{s,i} = W^{UK}_i c^{KV}_s$,$v_{s,i} = W^{UV}_i c^{KV}_s$
3. attention 与 MHA 同

这要每个 head 做一次小矩阵乘 → 仍有 overhead。

### 24.5.2 吸收技巧

观察 attention 分数(语义部分):

$$
q^{(C)\,\top}_{t,i} k^{(C)}_{s,i} = h_t^\top \big(W^{Q,C}_i\big)^\top W^{UK}_i\, c^{KV}_s
$$

定义新矩阵 $\tilde W^Q_i = (W^{UK}_i)^\top W^{Q,C}_i \in \mathbb{R}^{d_c \times d}$,**离线合并好**。推理时直接:

$$
\tilde q^{(C)}_{t,i} = \tilde W^Q_i h_t \in \mathbb{R}^{d_c}
$$

attention 分数:

$$
q^{(C)\,\top}_{t,i} k^{(C)}_{s,i} = \tilde q^{(C)\,\top}_{t,i} c^{KV}_s
$$

→ **完全在低维 $d_c$ 上做内积,无需还原 $k^{(C)}$**。

类似地 V 端的输出投影也可与 $W^{UV}$ 吸收。最终推理时:

- KV cache 只存 $(c^{KV}_t, k^{(R)}_t)$
- attention 全在低维空间运算
- 没有显式 $K_{n_h \times d_h}, V_{n_h \times d_h}$ 张量

### 24.5.3 训练 vs 推理

训练时通常显式构造 $k, v$(便于反向传播一致与 dense 实现统一);推理时切到吸收形式。两者数学等价,只是计算图重排。

---

## 24.6 KV cache 大小与压缩比

按 V3 配置(下节给完整数字),对照各 attention 变种 **每 token 每层 KV 字节**(bf16):

| 模型 / 变种 | 公式 | 单 token 字节 |
| --- | --- | --- |
| MHA(LLaMA-2-7B) | $2 \cdot 32 \cdot 128 \cdot 2$ | 16,384(16 KB) |
| GQA(LLaMA-3-70B,8 组) | $2 \cdot 8 \cdot 128 \cdot 2$ | 4,096(4 KB) |
| MQA(Falcon) | $2 \cdot 1 \cdot 128 \cdot 2$ | 512 |
| **MLA(V3)** | $(d_c + d_h^{(R)}) \cdot 2 = 576 \cdot 2$ | **1,152(约 1.1 KB)** |

V3 总共 61 层 → 单 token KV 总字节 ≈ 70 KB(MHA 同规模 ~ 1 MB)。

### 24.6.1 32k 上下文实测

| 模型 | KV cache(32k) |
| --- | --- |
| LLaMA-2-7B MHA | 16 GB |
| LLaMA-3-70B GQA | 10 GB |
| **DeepSeek-V3 MLA** | **~2 GB** |

V3 的 KV 比 LLaMA-3-70B 少 5×,且模型大近 10×。这就是 V3 长上下文 + 多并发的经济性来源。

### 24.6.2 128k 上下文

V3 在 128k 上下文下,KV cache 仍可控在十几 GB 量级,**单 H100 80GB 就能装下完整 KV + 部分模型权重(配合 MoE 切片)**,这是同规模 MHA 模型完全不敢想象的。

---

## 24.7 复杂度分析

### 24.7.1 算力(decode per step)

- 主要 cost = $W^{DKV} h_t$、$W^{KR} h_t$、Q 投影
- 加 Attention 与 $c^{KV}$ 的内积:$O(n \cdot d_c)$,其中 $n$ 是历史长度

总 decode FLOPs:$O(n d_c + d \cdot d_c)$,与 MHA 同阶但常数更小(因为 $d_c < d_k \cdot n_h$)。

### 24.7.2 显存带宽

decode 每 step 需要读所有历史 token 的 $c^{KV}, k^{(R)}$,**总带宽 $O(n \cdot d_c)$**,远小于 MHA 的 $O(n \cdot n_h \cdot d_k)$。

→ MLA 在 decode 带宽受限场景(§21.1)的优势比"算力 / 显存"两数字之比还更夸张。

### 24.7.3 训练算力

训练时显式 $k, v$,额外要算 $W^{UK} c, W^{UV} c$,FLOPs 略多于 MHA。但 $d_c \ll d$,代价 < 5%。**训练略贵,推理巨省**——典型的"花训练 cost 换推理 cost"设计,完美契合 inference-aware scaling(§17.4)。

---

## 24.8 完整代码实现

下面是简化版 MLA(单 batch,展示核心思想)。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 d_c: int = 512,
                 d_h_c: int = 128,
                 d_h_r: int = 64,
                 max_seq: int = 4096,
                 rope_base: float = 10000.0):
        super().__init__()
        self.h, self.d_c, self.d_h_c, self.d_h_r = n_heads, d_c, d_h_c, d_h_r
        # 下投影到潜变量
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)
        self.W_KR  = nn.Linear(d_model, d_h_r, bias=False)
        # 上投影到 K_C, V(每 head 各自)
        self.W_UK = nn.Linear(d_c, n_heads * d_h_c, bias=False)
        self.W_UV = nn.Linear(d_c, n_heads * d_h_c, bias=False)
        # Q 投影
        self.W_QC = nn.Linear(d_model, n_heads * d_h_c, bias=False)
        self.W_QR = nn.Linear(d_model, n_heads * d_h_r, bias=False)
        # 输出投影
        self.W_O = nn.Linear(n_heads * d_h_c, d_model, bias=False)

        # 预计算 RoPE
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, d_h_r, 2).float() / d_h_r))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
        self.register_buffer("sin", torch.cat([freqs.sin(), freqs.sin()], dim=-1))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rope(self, x, offset=0):
        T = x.size(-2)
        cos = self.cos[offset:offset+T]
        sin = self.sin[offset:offset+T]
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x: torch.Tensor, past_kv=None, offset: int = 0):
        # x: (B, T, d), past_kv: (c_kv_past, k_r_past) or None
        B, T, _ = x.shape
        c_kv = self.W_DKV(x)                          # (B, T, d_c)
        k_r  = self.W_KR(x).view(B, T, self.d_h_r)    # (B, T, d_h_r) 单份共享
        k_r  = self._rope(k_r, offset=offset)

        # 拼接历史
        if past_kv is not None:
            c_past, k_r_past = past_kv
            c_kv = torch.cat([c_past, c_kv], dim=1)
            k_r  = torch.cat([k_r_past, k_r], dim=1)

        # 展开 K_C, V(per head)
        N = c_kv.size(1)
        k_c = self.W_UK(c_kv).view(B, N, self.h, self.d_h_c)
        v   = self.W_UV(c_kv).view(B, N, self.h, self.d_h_c)
        # K = concat[K_C, K_R(broadcast 到所有 head)]
        k_r_b = k_r.unsqueeze(2).expand(B, N, self.h, self.d_h_r)
        k = torch.cat([k_c, k_r_b], dim=-1)           # (B, N, h, d_h)
        v_full = v                                    # V 只用 d_h_c 维(常见配置)

        # Q
        q_c = self.W_QC(x).view(B, T, self.h, self.d_h_c)
        q_r = self.W_QR(x).view(B, T, self.h, self.d_h_r)
        q_r = self._rope(q_r, offset=offset)
        q = torch.cat([q_c, q_r], dim=-1)             # (B, T, h, d_h)

        # Attention(因果掩码)
        # 转 [B, h, T, d_h]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v_full = v_full.transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        # causal: 当前 token 只能看到历史 N - T ... 当前
        if past_kv is None:                           # prefill 全因果
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v_full).transpose(1, 2).reshape(B, T, self.h * self.d_h_c)
        return self.W_O(out), (c_kv, k_r)
```

> 教学版,矩阵吸收优化、FP8 路径未展开。生产实现见 vLLM / DeepEP / SGLang 的 MLA kernel。

---

## 24.9 与 MHA / MQA / GQA 的对照

| 变种 | KV 单 token | 质量损失 | 工程复杂度 | 长上下文 |
| --- | --- | --- | --- | --- |
| MHA | $2 n_h d_k$ | 0 | 低 | 显存爆炸 |
| MQA | $2 d_k$ | 略 | 低 | 友好 |
| GQA | $2 d_k \cdot g$ | 极小 | 低 | 友好 |
| **MLA** | $d_c + d_h^{(R)}$ | 0(≈ MHA) | **高(decoupled RoPE)** | **最友好** |

MLA 的工程复杂度高(两路 K + 吸收),但**质量与 MHA 等同 / 略优**,这是它胜过 GQA 的关键。

---

## 24.10 V3 中的实际配置

DeepSeek-V3 默认值(技术报告 Table 2 / Appendix):

| 项 | 值 |
| --- | --- |
| 层数 $L$ | 61 |
| hidden $d$ | 7168 |
| heads $n_h$ | 128 |
| 每 head 语义维 $d_h^{(C)}$ | 128 |
| 每 head 位置维 $d_h^{(R)}$ | 64 |
| KV 潜变量维 $d_c$ | 512 |
| RoPE base | 10000(预训练 4k)+ YaRN 扩 128k |
| Q 端低秩压缩 | V2 用 1536,V3 移除(直接投影) |

→ 每 token 每层 KV 字节(bf16) = $(512 + 64) \times 2 = 1{,}152$ B。

→ 总 KV(128k 上下文,61 层):$128000 \times 61 \times 1152 \approx 8.4$ GB。

同规模 MHA 估算(若 V3 用 MHA):$128 \cdot 128 \cdot 2 \cdot 61 \cdot 128000 \cdot 2 \approx 480$ GB → 完全不可行。

**结论:MLA 是 V3 128k 上下文与商业经济性的物理前提**。

---

## 🎨 24.10.A 通俗类比合集

### 类比一: 拼装家具 → 折叠家具 📦

> KV cache 是"已经组装好的家具",占地方;MLA 把它做成"折叠家具"。

- **MHA** = 每张椅子全装好放角落,占地巨大
- **GQA** = 几个椅子共用一组零件,稍省
- **MLA** = 家具压成扁平包装 $c_t$(潜变量),用的时候用一套**统一上料板** ($W^{UK}, W^{UV}$) 展开
- **RoPE 解耦** = 折叠家具不能在折叠状态贴座位号(位置编码),所以**另外发一张"座位卡"** $k^{(R)}$
- **矩阵吸收** = 商家偷偷把"开包+组装"两步合一,客户拿出来就能用,不真的展开整张椅子

### 类比二: 索引型数据库 🗃️

> 把每条历史记录全文存太占空间 → 改成"索引 + 模板复原"。

- **MHA** = 全文档逐字存
- **MLA** = 只存"主键 $c_t$"(短)+ "模板矩阵 $W^{UK/UV}$"(全局一份,共享) → 查询时按主键展开
- **decoupled RoPE** = 时间戳单独维护(不进主键),否则主键得每秒重生成
- **吸收** = 查询 SQL 直接在索引里跑,不真的物化整行数据

### 类比三: 缩略图 + 原图按需展开 🖼️

> 网盘里 1 万张照片,本地只存缩略图。

- **缩略图 = $c_t$ 潜变量**(小、足够检索)
- **服务端的原图模板($W^{UK}$)**:每次需要,服务端用模板把缩略图还原
- **decoupled RoPE** = 拍摄时间 / GPS 单独存在元数据,不揉进图本身
- **矩阵吸收** = 检索操作直接在缩略图域做相似度,完全不需要 fullsize 原图

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 📦 折叠家具 | **压缩 + 展开** | $W^{UK}$ 上投影 |
| 🗃️ 索引 | **主键 + 模板** | $c_t$ 的角色 |
| 🖼️ 缩略图 | **延迟具体化** | 矩阵吸收 |

**记忆口诀**:
> **"语义压成潜变量,位置走单独通道,矩阵吸收省展开,KV 一秒变白菜。"**

---

## 24.11 常见疑问

### Q1: MLA 是不是把 KV 量化到 INT4 一样?
**A**: 不是。量化是"用更少 bit 存原向量",信息无损是有限的;MLA 是"换一个更紧凑的表征空间(低秩潜变量)",**信息论上有损但通过模型学习重投影几乎无质量代价**。两者正交,可同时用(MLA + INT8 KV 进一步压)。

### Q2: 为什么不直接训练一个 $d_k$ 很小的 MHA?
**A**: 直接砍 $d_k$ 会损质量;MLA 等价于保留 $d_h^{(C)}$ 但**多 head 共享同一组语义潜变量** $c_t$,信息利用率更高。

### Q3: MLA 兼容 Flash Attention 吗?
**A**: 兼容,但需要 MLA 特化的 kernel(语义部分 + 位置部分两个 score 累加)。vLLM / DeepEP 已经实现。社区开源工具(FlashMLA)2024 年底发布。

### Q4: 训练时算 cost 比 MHA 高吗?
**A**: 略高(< 5%)。多了 $W^{UK}, W^{UV}, W^{KR}$ 的投影。但训练 token 通常远多于推理,所以"训练贵一点点 + 推理省巨大"的总账非常划算。

### Q5: Q 端为什么 V3 取消了低秩压缩?
**A**: Q 不进入 KV cache,**压缩只省训练时一次 forward,收益小**;反而带来工程复杂度。V3 简化:Q 直接对 $h_t$ 投影,K/V 才用 MLA。

---

## 24.12 本章小结

### 核心公式

```text
c_t   = W_DKV · h_t               (低秩潜变量,所有 head 共享)
k_C_i = W_UK_i · c_t              (语义 K,每 head)
k_R   = RoPE(W_KR · h_t)          (位置 K,所有 head 共享)
k_i   = [k_C_i ; k_R]
v_i   = W_UV_i · c_t
score = q_C · k_C + q_R · k_R     (两路内积相加)
```

**KV cache 内容**:$(c_t, k_R)$,所有 head 共享。

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **思想** | 低秩 KV + decoupled RoPE |
| **压缩比** | ~50-100× vs MHA |
| **质量** | 与 MHA 等同 |
| **推理优化** | 矩阵吸收,attention 在低维空间算 |
| **配套** | 训练略贵,推理巨省 |

### 一句话总结

> **MLA 把 KV cache 压成低秩潜变量 $c_t$ + 一个共享的位置专用 $k^{(R)}$,通过 decoupled RoPE 解决"压缩与位置编码冲突",通过矩阵吸收让推理 attention 直接在低维空间算——单 token KV 字节比 MHA 降 50× 以上,是 DeepSeek-V3 实现 671B 模型 + 128k 上下文 + 商业经济性的物理前提。**

---

## 24.13 延伸阅读

### 必读论文

1. DeepSeek-AI, 2024. *"DeepSeek-V2"*(MLA 首次提出). [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
2. DeepSeek-AI, 2024. *"DeepSeek-V3 Technical Report"*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
3. Su et al., 2021. *"RoFormer / RoPE"*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

### 进阶资源

4. FlashMLA(DeepSeek 官方开源 kernel)
5. vLLM MLA backend 源码
6. Anthropic / Mistral 等的 MLA-like 复现讨论
7. *"Comparing MHA / MQA / GQA / MLA"* 社区分析博客

---

## 24.14 实战练习

### 练习 1 (★)

用 §24.8 的 `MLA` 类替换 §10.6 GPTMini 的 attention,在 tiny shakespeare 上训练 1000 步,验证 loss 与 MHA 大体一致。

### 练习 2 (★★)

实现矩阵吸收推理路径:把 $W^Q (W^{UK})^\top$ 离线合并,推理时直接在 $d_c$ 维做内积。对比朴素 forward 的速度与显存。

### 练习 3 (★★)

按 §24.6 公式,写函数 `mla_kv_bytes(L, n, d_c, d_h_r, bytes_per_elem)`,与 MHA / GQA 对比,复现 §24.10 的 V3 数字。

### 练习 4 (★★★)

读 vLLM 的 `mla.py`,梳理它如何在 PagedAttention 上存 $(c, k_r)$;画一份 KV page 布局图。

### 练习 5 (★★★)

实现一个 ablation:固定其他,**MLA 设置 $d_h^{(R)} = 0$(完全无 RoPE)** 训练对比;再做 $d_h^{(R)} = 64$,验证位置信息确实在 $k^{(R)}$ 这条通道里。

---

## 章节交叉引用

- 前置: [§07 Self-Attention](../Part_II_Transformer架构/07_Self_Attention机制.md), [§06 RoPE](../Part_II_Transformer架构/06_Embedding与位置编码.md), [§19 KV Cache](../Part_IV_推理与部署/19_KV_Cache与显存管理.md)
- 后续: [§26 V3 训练曲线](26_V3_训练曲线与工程.md), [§27 V3 vs LLaMA-3](27_V3_vs_LLaMA3.md)
- 相关: [§01 线性代数 SVD](../Part_I_数学与基础/01_线性代数与矩阵运算.md), [§21 推理优化](../Part_IV_推理与部署/21_推理优化.md)

---

*下一章: §25 DeepSeekMoE*
