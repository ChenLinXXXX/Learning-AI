# 10 Transformer Block 整体(从 Token 到 Logits:Decoder-Only LLM 完整数据流)

> **章节定位**: Part II 收尾合成章,把 §05–§09 的零件拼装成一个可运行的 GPT
> **预计阅读时间**: 70-90 分钟
> **难度**: ★★★★☆
> **前置知识**: §05–§09 已完整学习
> **后续依赖**: §13 训练循环、§18 推理流程、§24 MLA(本章是 MLA 的基础参照)

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [10.1 现代 LLM 总体架构:Decoder-Only](#101-现代-llm-总体架构decoder-only)
- [10.2 单个 Transformer Block 内部数据流](#102-单个-transformer-block-内部数据流)
- [10.3 从 Token 到 Logits 的完整 forward](#103-从-token-到-logits-的完整-forward)
- [10.4 参数量公式与"模型体检表"](#104-参数量公式与模型体检表)
- [10.5 训练目标:Next-Token Prediction](#105-训练目标next-token-prediction)
- [10.6 完整代码实现:50 行 GPT](#106-完整代码实现50-行-gpt)
- [10.7 复杂度与显存预算](#107-复杂度与显存预算)
- [10.8 三大主流变体对比:GPT / LLaMA / DeepSeek](#108-三大主流变体对比gpt--llama--deepseek)
- [10.9 可解释性视角:Residual Stream](#109-可解释性视角residual-stream)
- [🎨 10.9.A 通俗类比合集](#-109a-通俗类比合集)
- [10.10 常见疑问](#1010-常见疑问)
- [10.11 本章小结](#1011-本章小结)
- [10.12 延伸阅读](#1012-延伸阅读)
- [10.13 实战练习](#1013-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

前 5 章分别讲了"零件":Tokenization(§05)、Embedding 与位置编码(§06)、Self-Attention(§07)、FFN(§08)、归一化与残差(§09)。本章把这些零件**按现代 LLM 的标准拼接**成一个 Decoder-Only Transformer,完整呈现:

- 整体架构与 forward 数据流
- 每个张量的形状与含义
- 参数量、FLOPs、显存预算的工程估算公式
- Next-Token Prediction 的训练目标与 cross-entropy 实现
- GPT / LLaMA / DeepSeek 三大代表的设计差异
- Residual Stream 视角下的可解释性入门

读完本章你应该能用 100 行以内的 PyTorch 写一个能训练能采样的迷你 GPT,并对照真实 LLM 配置反推参数量。

---

## 学习目标

完成本章后,你能够:

- ✅ 画出 Decoder-Only Transformer 的完整方块图 (§10.1)
- ✅ 列出从 token ID 到 logits 每一步的张量形状 (§10.3)
- ✅ 用闭式公式估算总参数量(给定 $d, h, N, V, d_{\text{ff}}$) (§10.4)
- ✅ 解释 Next-Token Prediction 的 loss 形式与 shift 操作 (§10.5)
- ✅ 写一个 < 100 行的 GPT-mini,可以训练玩具数据 (§10.6)
- ✅ 比较 GPT-3 / LLaMA-2 / DeepSeek-V3 在结构上的核心差异 (§10.8)

---

## 10.1 现代 LLM 总体架构:Decoder-Only

### 10.1.1 三种 Transformer 架构家族

| 架构 | 代表 | 用途 |
| --- | --- | --- |
| Encoder-Only | BERT、RoBERTa | 理解 / 表示学习 |
| Encoder-Decoder | T5、BART、原版 Transformer | 翻译 / 摘要 |
| **Decoder-Only** | **GPT 系列、LLaMA、DeepSeek、Qwen 等所有现代 LLM** | **生成 / 通用 chat** |

Decoder-Only 之所以胜出,原因有三:

1. **任务统一**:用 prompt + 续写就能做几乎所有 NLP 任务,无需为每任务设计 encoder/decoder
2. **训练简单**:只用一个 LM loss
3. **KV cache 友好**:自回归生成天然适配增量解码

### 10.1.2 总体方块图

```text
Token IDs (B, T)
     │
     ▼
[Embedding]  →  (B, T, d_model)
     │
     ▼  ←─────── 残差主干 (residual stream)
┌────────────────────────────┐
│ Block 1                    │
│  ├─ RMSNorm                │
│  ├─ Multi-Head Attention   │ (with RoPE on Q,K, causal mask)
│  ├─ + (残差)               │
│  ├─ RMSNorm                │
│  ├─ FFN (SwiGLU)           │
│  └─ + (残差)               │
└────────────────────────────┘
     │
     ▼
   …  (重复 N 个 Block)
     │
     ▼
[Final RMSNorm]
     │
     ▼
[LM Head]  →  (B, T, V)  logits
     │
     ▼
[Softmax / Sampling]
```

### 10.1.3 关键设计原则

- 主干永远是**残差流**,所有信息流过它
- 每个子层"先归一化,再处理,再加回主干"
- 位置信息通过 **RoPE 加在 Q、K 上**(不进 embedding)
- 最后用 **Final RMSNorm + LM Head** 把主干向量翻译成词表 logits

---

## 10.2 单个 Transformer Block 内部数据流

写成代数形式,Block $l$ 的运算:

$$
\begin{aligned}
\tilde h^{(l)} &= h^{(l-1)} + \text{Attn}\!\big(\text{RMSNorm}(h^{(l-1)})\big) \\
h^{(l)} &= \tilde h^{(l)} + \text{FFN}\!\big(\text{RMSNorm}(\tilde h^{(l)})\big)
\end{aligned} \tag{10.1}
$$

其中 $h^{(0)} = E[\text{tokens}]$(embedding 后的隐藏状态)。

**关键张量形状**(batch=B,序列长 T,模型维度 d):

| 张量 | 形状 |
| --- | --- |
| $h^{(l)}$ | $(B, T, d)$ |
| $Q, K, V$(MHA 内部) | $(B, h, T, d_k)$ |
| Attention 分数 $A$ | $(B, h, T, T)$ |
| FFN 隐藏 | $(B, T, d_{\text{ff}})$ |
| Logits | $(B, T, V)$ |

---

## 10.3 从 Token 到 Logits 的完整 forward

```text
1. tokens: (B, T)            int64
2. h = Embedding(tokens)     (B, T, d)
3. for l in 1..N:
       h = Block_l(h)        (B, T, d)
4. h = FinalRMSNorm(h)       (B, T, d)
5. logits = LMHead(h)        (B, T, V)
6. probs  = softmax(logits)  (训练时常省,直接交叉熵)
```

每一步都是**纯函数式**,可以单元测试。

---

## 10.4 参数量公式与"模型体检表"

记 $d = d_{\text{model}}$,$d_{\text{ff}}$、词表 $V$、层数 $N$、bias 默认无。

| 组件 | 参数量 |
| --- | --- |
| Embedding | $V \cdot d$ |
| 每层 Attention(MHA) | $4 d^2$($W_Q, W_K, W_V, W_O$) |
| 每层 FFN(标准 2 矩阵) | $2 d \cdot d_{\text{ff}} = 8 d^2$(当 $d_{\text{ff}}=4d$) |
| 每层 SwiGLU FFN(3 矩阵) | $3 d \cdot d_{\text{ff}} \approx 8 d^2$(当 $d_{\text{ff}}=\tfrac{8}{3}d$) |
| 每层 Norm | $\approx 2d$(两组 $\gamma$) |
| Final Norm + LM Head | $d + V d$(LM Head 与 Embedding 可绑) |

**总参数(SwiGLU、tie weights)**:

$$
P \approx V \cdot d + N \cdot (4 d^2 + 8 d^2) = V d + 12 N d^2 \tag{10.2}
$$

### 10.4.1 实例核对

| 模型 | $N$ | $d$ | $V$ | 公式估算 | 公布值 |
| --- | --- | --- | --- | --- | --- |
| GPT-2 small | 12 | 768 | 50257 | ≈ 124 M | 124 M ✅ |
| LLaMA-2 7B | 32 | 4096 | 32000 | ≈ 6.5 B | 6.7 B ✅ |
| LLaMA-2 13B | 40 | 5120 | 32000 | ≈ 12.7 B | 13.0 B ✅ |
| LLaMA-2 70B | 80 | 8192 | 32000 | ≈ 64 B(GQA 修正后) | 70 B(用 GQA 节约) |

注意 LLaMA-2-70B 用 **GQA**(Grouped Query Attention,$h_{\text{kv}}=8$),Attention 部分参数比标准 MHA 少;公式 10.2 是 MHA 近似,大模型需按 GQA/MLA 修正(详见 §24)。

### 10.4.2 一份 50 秒"体检"

拿到一份模型卡(`config.json`),按顺序问:

1. $N$, $d$, $h$, $d_{\text{ff}}$, $V$ 是多少?
2. KV head 数:MHA / GQA / MLA?
3. Norm 是 LN 还是 RMSNorm?Pre 还是 Post?
4. 位置编码:Sinusoidal / RoPE / ALiBi?RoPE base 多大?
5. Activation:GELU / SwiGLU?
6. Embedding 与 LM head 是否 tie?

回答完这 6 题,你就基本掌握了任何 LLM 的结构指纹。

---

## 10.5 训练目标:Next-Token Prediction

### 10.5.1 损失函数

给定序列 $x_1, x_2, \ldots, x_T$,模型在每个位置输出 logits $z_t \in \mathbb{R}^V$,目标是预测**下一个 token** $x_{t+1}$:

$$
\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log p_\theta(x_{t+1} \mid x_{\leq t}) \tag{10.3}
$$

其中 $p_\theta(\cdot \mid x_{\leq t}) = \text{softmax}(z_t)$。这就是 **Cross-Entropy + 自回归** 的标准设置。

### 10.5.2 实现细节:Shift

```python
logits = model(tokens)            # (B, T, V)
loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, V),   # 取前 T-1 个位置的预测
    tokens[:, 1:].reshape(-1),          # 与右移一位的标签对齐
)
```

**两个易错点**:

- 必须切片 `[:-1]` 与 `[1:]`,否则把"自己预测自己"算进去
- Causal Mask 由 Attention 内部保证;loss 不需要再 mask 一次

### 10.5.3 Cross-Entropy 与 Perplexity 的关系

$$
\text{PPL} = \exp(\mathcal{L}) \tag{10.4}
$$

PPL 是"平均每步在多少 token 中犹豫"的直觉度量。$V = 50257$,随机猜 PPL ≈ 50257;训练好的 GPT-2 small 在 WikiText 上 PPL ≈ 25 → 大约把"50257 选 1"压缩到"25 选 1"。

---

## 10.6 完整代码实现:50 行 GPT

下面整合 §05–§09 所有零件。可直接训练玩具数据。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# (Re-use) RMSNorm / RoPE / MultiHeadAttention / SwiGLUFFN  —— 见 §06、§07、§08、§09

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads, causal=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn (self.norm2(x))
        return x


class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_heads=6,
                 d_ff=None, n_layers=6, max_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm_f  = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.tok_emb.weight
        self.max_len = max_len

    def forward(self, idx, targets=None):
        # idx: (B, T)
        x = self.tok_emb(idx)                  # (B, T, d)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)               # (B, T, V)

        if targets is None:
            return logits, None
        # Next-token loss with shift
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=64, temperature=1.0, top_k=None):
        for _ in range(max_new):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
```

> Attention 内部应集成 §06 的 RoPE;为聚焦在 Block 拼接逻辑,代码留了占位。完整 100 行实现请参考 Karpathy 的 nanoGPT。

---

## 10.7 复杂度与显存预算

### 10.7.1 单步 FLOPs(粗算)

主导项:

$$
\text{FLOPs} \approx 2 P \cdot T \tag{10.5}
$$

其中 $P$ 是模型参数量,$T$ 是序列长度。这是经验法则——"训练 1 token 大约消耗 2P 次乘加"。

### 10.7.2 训练总算力

训练 $D$ 个 token 需要算力约:

$$
C \approx 6 P \cdot D \tag{10.6}
$$

(2P forward + 2P backward + 2P optimizer)。**Chinchilla scaling law** 建议 $D \approx 20 P$,即"训练 token 数应约为参数量的 20 倍"(详见 §17)。

### 10.7.3 显存预算

训练时需要的显存大致为(bf16 + AdamW):

| 项 | 大小 |
| --- | --- |
| 权重 | $2P$ 字节 |
| 梯度 | $2P$ |
| 优化器状态(Adam) | $4P$ |
| Activations | $\propto B \cdot N \cdot d \cdot T$(可通过激活检查点压缩) |

例:7B 模型训练显存 ≈ 16 × 7B = 112 GB,所以必须**多卡 + ZeRO / PP / TP**(详见 §14)。

### 10.7.4 推理 vs 训练

推理时只需 forward + KV cache,显存远小:

- 7B 模型(bf16):权重 14 GB;KV cache(MHA,32 层,128 序列)≈ 0.5 GB;**单 GPU 24 GB 即可推理**
- 70B 模型:权重 140 GB → 必须多卡 + 量化(详见 §22)

---

## 10.8 三大主流变体对比:GPT / LLaMA / DeepSeek

| 维度 | GPT-3 (175B) | LLaMA-2 (7B/13B/70B) | DeepSeek-V3 (671B / 37B 激活) |
| --- | --- | --- | --- |
| **Norm** | LayerNorm,Pre-Norm | RMSNorm,Pre-Norm | RMSNorm,Pre-Norm |
| **Activation** | GELU | SwiGLU | SwiGLU |
| **Position** | 学习式绝对 | RoPE | RoPE(decoupled,§24) |
| **Attention** | MHA | MHA(7B/13B)/ GQA(70B) | **MLA**(§24) |
| **FFN** | Dense | Dense | **MoE**(细粒度 + 共享专家,§23) |
| **Tie weights** | ✅ | ❌ | ❌ |
| **训练精度** | fp16 | bf16 | **FP8 + bf16 主流通信**(§13) |
| **上下文** | 2k | 4k(LLaMA-2 → 8k,LLaMA-3 → 128k) | 4k 预训练 + YaRN 扩到 128k |

**演进脉络**(2020 → 2025):

1. LayerNorm → RMSNorm
2. GELU → SwiGLU
3. 学习式位置 → RoPE → RoPE + YaRN
4. MHA → GQA → MLA(减 KV cache)
5. Dense → MoE(减激活参数)
6. fp16 → bf16 → FP8(减算力 / 通信)

Part II 涵盖前 3 项,Part V(DeepSeek 专题)讲后 3 项。

---

## 10.9 可解释性视角:Residual Stream

Anthropic 的 *"A Mathematical Framework for Transformer Circuits"* 提出:把残差主干 $h$ 视为一条 **Residual Stream**:

- 每个子层从主干**读取**($\text{Norm}(h)$)信息
- 经过计算后把结果**写回**主干($h \mathrel{+}= f$)

主干 = 一根"信息总线",N 层 Block 在它上面叠加各种"读 / 写"操作。这一视角让"诱导头(induction heads)"、"事实回忆电路"等可解释性发现成为可能。

**实用启示**:

- 主干维度 $d$ 必须足够大,以容纳所有子层往里写的信息——这就是为什么大模型 $d$ 也要随之增大
- 残差流分析可以解释"哪些层负责哪些任务"

---

## 🎨 10.9.A 通俗类比合集

### 类比一: 工厂里的"主传送带 + 多道工序" 🏭

> 一条主传送带从 A 端到 B 端;每隔一段就有一道工序站,从带上取一份半成品加工后**叠加回去**。

- **主传送带** = Residual Stream(残差主干)
- **每道工序站** = 一个 Transformer Block(Attention + FFN)
- **取下来** = `Norm(x)`(标准化才能进工序)
- **加工后叠加** = `x + f(Norm(x))`(残差加法)
- **每个工序站只能改一点点** = 单层贡献相对较小,模型靠 N 层累积

经过 32 / 100 个工序站,主传送带运出的就是"高质量成品向量",最后由 LM Head 翻译成"应该输出哪个 token"。

### 类比二: 城市供水主管道 + N 个净水站 💧

> 自来水从源头一路流向城市,中途经过多个净水站。每个站抽走一点水检测处理,再把处理后的水"加回主管道"。

- **主管道** = Residual Stream
- **净水站** = Block
- **入站前过滤(Norm)**:统一水压、水质再处理
- **不抽完所有水(残差)**:即使某站出故障,主管道也不断流
- **管道直径 (d)**:必须足够大,否则净水站搬不动这么多信息

### 类比三: 学校教学楼里的"知识管道课表" 📚

> 想象一名学生 token 从早上 8 点入校,每节课(Block)是"自学一段 + 做题一段",形成 32 节课的完整学习日。

- **学生进校** = Embedding
- **每节课的"自学时间"** = Attention(向同学借笔记,跨 token 互动)
- **每节课的"做题时间"** = FFN(独立内化,token 内深加工)
- **下课铃 + 错题本** = 残差(把今日所得加在昨天的基础上,不推翻)
- **校服(Norm)** = 每节课前先统一着装,情绪稳定再上课
- **放学测验** = LM Head + Softmax(把一天所学翻译成对"下一个明天该干啥"的预测)

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🏭 工厂传送带 | **主干 + 累加** | Residual Stream 视角 |
| 💧 供水管道 | **稳定流量** | Norm 与 Residual 的协同 |
| 📚 学习日 | **训练目标** | Next-Token 与 LM Head |

**记忆口诀**:
> **"主干传送带,每层来加料;先读 Norm 入,再 + 写出去。"**

---

## 10.10 常见疑问

### Q1: 为什么 Decoder-Only 现在压倒 Encoder-Decoder?
**A**: 工程上更通用(prompt + 续写覆盖所有任务)、训练目标更简单(只 LM loss)、KV cache 适配自回归生成。理论上两者表达力相当,工程优势让 Decoder-Only 胜出。

### Q2: 为什么不直接对最后一个 token 算 loss,而要对每个位置算?
**A**: 因为 causal mask 保证位置 $t$ 只看过 $\leq t$,每个位置都是一次独立的"next-token"预测——把整段序列上的 T-1 次预测全部纳入 loss,等于一个 batch 包含 T-1 个训练样本,数据效率最大化。

### Q3: 序列长度 T 在训练和推理时一定一样吗?
**A**: 不一定。训练通常固定 $T$(如 4096)便于批处理;推理时变长,KV cache 累积,**自然外推**前提是位置编码支持。RoPE + YaRN(§06)就是为解决"训练短、推理长"。

### Q4: bias 全去掉了,模型能学到偏置吗?
**A**: 能。归一化层的 $\gamma$(以及 LayerNorm 的 $\beta$)提供了仿射自由度;Attention/FFN 自身的可学权重也能在大量数据下吸收"偏置"。实证显示去 bias 训练更稳定、参数更省。

### Q5: 怎么直观判断一个 LLM 的"质量天花板"?
**A**: 看 **scaling triple**:(参数量 $P$,训练 token $D$,数据质量)。Chinchilla 建议 $D \approx 20 P$,低于此值意味着"训练不足";高出很多则边际收益递减。详见 §17。

---

## 10.11 本章小结

### 核心公式

**单 Block**:

$$
\begin{aligned}
\tilde h &= h + \text{Attn}(\text{RMSNorm}(h)) \\
h' &= \tilde h + \text{FFN}(\text{RMSNorm}(\tilde h))
\end{aligned}
$$

**总参数(SwiGLU + tie)**:

$$
P \approx V d + 12 N d^2
$$

**Next-Token Loss**:

$$
\mathcal{L} = -\frac{1}{T-1}\sum_{t=1}^{T-1} \log p_\theta(x_{t+1}\mid x_{\le t})
$$

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **架构** | Decoder-Only,N 个 Block 堆叠 |
| **数据流** | Embedding → N×(Norm+Attn+残差 / Norm+FFN+残差)→ FinalNorm → LMHead |
| **目标** | Next-Token Prediction,Cross-Entropy |
| **算力** | 单步 ≈ 2P·T,训练 ≈ 6PD |
| **可解释性** | Residual Stream 是信息总线 |

### 一句话总结

> **现代 LLM 是 N 层"Pre-RMSNorm + Self-Attention + 残差 + FFN(SwiGLU) + 残差"的 Decoder-Only Transformer,以 Next-Token Prediction 为唯一训练目标;参数主体是 FFN(2/3),长上下文瓶颈在 Attention,后续 Part V 通过 MLA + MoE 各打一拳。**

---

## 10.12 延伸阅读

### 必读论文

1. Radford et al., 2018. *"Improving Language Understanding by Generative Pre-Training"*(GPT-1,Decoder-Only 范式确立)
2. Brown et al., 2020. *"Language Models are Few-Shot Learners"*. NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)(GPT-3)
3. Touvron et al., 2023. *"LLaMA"*. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
4. DeepSeek-AI, 2024. *"DeepSeek-V3 Technical Report"*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

### 进阶论文

5. Elhage et al., 2021. *"A Mathematical Framework for Transformer Circuits"*. Anthropic.(Residual Stream 概念)
6. Hoffmann et al., 2022. *"Training Compute-Optimal Large Language Models"*(Chinchilla)
7. Karpathy. *nanoGPT* repo & *"Let's build GPT: from scratch"* YouTube

### 推荐资源

- The Illustrated GPT-2 / The Illustrated Transformer
- HuggingFace `transformers` 源码 `modeling_llama.py`(对照本章公式逐行阅读)

---

## 10.13 实战练习

### 练习 1 (★)

把 §10.6 的 `GPTMini` 实际跑起来:用 "tiny shakespeare" 数据训练 1000 步,采样生成一段文本。

### 练习 2 (★)

修改 `GPTMini`,把 `tie_weights` 关掉,对比同 step 下 loss 与生成质量。

### 练习 3 (★★)

实现 §10.4 中"50 秒体检"脚本:读一个 HuggingFace `config.json`,自动输出结构指纹与公式估算的参数量,与 `model.num_parameters()` 比对误差。

### 练习 4 (★★)

把 `GPTMini` 改成 GQA 模式(KV head 数 = 2),验证 §10.4 中关于 70B 模型用 GQA 节约参数的论断。

### 练习 5 (★★★)

实现一个最小可运行的 KV cache + 增量解码,使 `generate` 的 wall time 不再随历史长度二次增长。这是 Part IV §19 的预热练习。

### 练习 6 (★★★)

绘制 Residual Stream 可视化:对训练好的 `GPTMini`,采样一段输入,记录每层 Block 进入前 / 写出后的主干向量,做 PCA 投影观察"信息累积"过程。

---

## 章节交叉引用

- 前置: 全部 §05–§09(本章是 Part II 的合成章)
- 后续: [§13 训练循环与混合精度](../Part_III_训练原理/13_训练循环与混合精度.md), [§18 推理流程](../Part_IV_推理与部署/18_推理流程.md)
- 相关: [§14 分布式训练](../Part_III_训练原理/14_分布式训练.md)(70B+ 模型如何切分本章模型), [§17 Scaling Law](../Part_III_训练原理/17_Scaling_Law.md), [§24 MLA](../Part_V_DeepSeek专题/24_MLA深度解析.md)(在本章基础上改 Attention)

---

*下一部分: Part III 训练原理(从数据到参数,模型如何"学会")*
