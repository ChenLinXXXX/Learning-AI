# 08 FFN 与激活函数(从 ReLU 到 SwiGLU 到 MoE)

> **章节定位**: Part II 核心章节,Transformer Block 的"另一半"
> **预计阅读时间**: 60-90 分钟
> **难度**: ★★★☆☆
> **前置知识**: §01 线性代数(矩阵乘法、逐元素运算), §07 Self-Attention(Block 上下文)
> **后续依赖**: §09(归一化)、§10(Block 整体)、§17(Scaling Law)、§23(DeepSeekMoE)

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [8.1 历史背景:从 MLP 到 Transformer FFN](#81-历史背景从-mlp-到-transformer-ffn)
- [8.2 标准 FFN:Position-wise Feed-Forward](#82-标准-ffnposition-wise-feed-forward)
- [8.3 激活函数演进:ReLU → GELU → SwiGLU](#83-激活函数演进relu--gelu--swiglu)
- [8.4 门控线性单元(GLU)家族](#84-门控线性单元glu家族)
- [8.5 为什么 FFN 占了模型大部分参数?](#85-为什么-ffn-占了模型大部分参数)
- [8.6 完整代码实现](#86-完整代码实现)
- [8.7 复杂度分析](#87-复杂度分析)
- [8.8 MoE:把 FFN 拆成专家](#88-moe把-ffn-拆成专家)
- [8.9 与 Attention 的角色对比](#89-与-attention-的角色对比)
- [🎨 8.9.A 通俗类比合集](#-89a-通俗类比合集)
- [8.10 常见疑问](#810-常见疑问)
- [8.11 本章小结](#811-本章小结)
- [8.12 延伸阅读](#812-延伸阅读)
- [8.13 实战练习](#813-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

Transformer Block 由两个核心子层组成:Self-Attention(§07)和 **Feed-Forward Network(FFN)**。如果说 Attention 负责"token 之间互相看",FFN 就负责"每个 token 独立做深加工"。本章系统梳理 FFN 的演进:

- **结构演进**: 标准两层 MLP → 门控 FFN(GLU)→ 稀疏 FFN(MoE)
- **激活演进**: ReLU → GELU → SwiGLU(LLaMA / DeepSeek 现代默认)
- **参数占比**: 主流 LLM 中,**约 2/3 参数都在 FFN**,这也是 MoE 收益的根源

本章给出严谨定义、SwiGLU 的代数推导、PyTorch 实现、参数量与 FLOPs 计算,并铺垫到 §23 DeepSeekMoE。

---

## 学习目标

完成本章后,你能够:

- ✅ 写出标准 FFN 的完整公式 (公式 8.1)
- ✅ 解释 SwiGLU 相对 ReLU/GELU 的优势 (§8.3.3)
- ✅ 推导 SwiGLU 的"三矩阵"参数量为何要除以 2/3 (§8.4.3)
- ✅ 计算给定 $d_{\text{model}}$、$d_{\text{ff}}$ 下 FFN 的参数量与 FLOPs (§8.5、§8.7)
- ✅ 用 30 行 PyTorch 实现 SwiGLU FFN (§8.6)
- ✅ 一句话说出 MoE 与 Dense FFN 在推理时的根本区别 (§8.8)

---

## 8.1 历史背景:从 MLP 到 Transformer FFN

### 8.1.1 MLP 是神经网络的"通用零件"

多层感知机(MLP)定义为:

$$
\text{MLP}(x) = W_2 \,\sigma(W_1 x + b_1) + b_2 \tag{8.1}
$$

其中 $\sigma$ 是非线性激活,$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$、$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$。

**通用近似定理**(Hornik, 1989):足够宽的两层 MLP 可以逼近任意连续函数。这是 FFN 在 Transformer 中作为"非线性表达力来源"的理论依据。

### 8.1.2 Transformer 中的 FFN 有什么特别?

Vaswani et al.(2017)在 Transformer Block 中放了一个 **Position-wise FFN**:

> "Position-wise" 的意思是 **对序列中每个位置独立应用同一个 MLP**,位置之间不共享信息。

这与卷积 / Attention 形成互补:

| 子层 | 跨 token 交互? | 非线性? | 角色 |
| --- | --- | --- | --- |
| Self-Attention | ✅ 强交互 | 弱(softmax) | "看其他位置" |
| FFN | ❌ 完全独立 | ✅ 强非线性 | "深加工自己" |

**关键洞察**: Attention 提供 token 间通信通道,FFN 提供逐 token 的"思考深度"。两者缺一不可。

### 8.1.3 一个常被忽略的事实

主流 LLM 中,FFN **不是配角**,而是参数主体:

- LLaMA-2-7B: 总参数 6.7B,FFN 约 **4.4B(≈ 66%)**
- GPT-3 175B: FFN 约 **117B(≈ 67%)**

后面 §8.5 会给出推导。这也直接催生了 MoE 这种"只激活部分 FFN"的稀疏架构。

---

## 8.2 标准 FFN:Position-wise Feed-Forward

### 8.2.1 定义

给定输入 $x \in \mathbb{R}^{d_{\text{model}}}$:

$$
\text{FFN}(x) = W_2 \,\sigma(W_1 x + b_1) + b_2 \tag{8.2}
$$

- $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ — **上投影 (up projection)**,先把维度放大
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ — **下投影 (down projection)**,再压回原维度
- $\sigma$ — 激活函数(原版用 ReLU)

通常 $d_{\text{ff}} = 4 \cdot d_{\text{model}}$,所以也叫"**先扩4倍,再压回**"。

### 8.2.2 为什么是 4 倍?

Vaswani 等人没给严格理由,这是经验值。直觉解释:

- 太窄($d_{\text{ff}} = d_{\text{model}}$):表达力受限,与线性层差不多
- 太宽($d_{\text{ff}} = 16 d_{\text{model}}$):参数爆炸,边际收益递减
- 4 倍在"表达力 vs 参数效率"之间取得平衡,后续大量实验确认这是甜点(参考 Shazeer 2020、Chinchilla 2022)

### 8.2.3 Position-wise 的工程含义

在实现上,把 $x$ 看作形状 $[\text{batch}, \text{seq\_len}, d_{\text{model}}]$ 的张量,FFN 直接对最后一维做 $\text{nn.Linear}$ 即可,**不需要循环每个位置**——PyTorch 的 Linear 自动 broadcast。

```python
# x: [B, T, d_model]
h = W1(x)         # [B, T, d_ff]
h = activation(h) # [B, T, d_ff]
y = W2(h)         # [B, T, d_model]
```

---

## 8.3 激活函数演进:ReLU → GELU → SwiGLU

### 8.3.1 ReLU(原版 Transformer)

$$
\text{ReLU}(x) = \max(0, x) \tag{8.3}
$$

优点:简单、稀疏、梯度不消失;缺点:**负半轴梯度恒为 0**,容易出现"死神经元"。

### 8.3.2 GELU(BERT / GPT-2 / GPT-3 默认)

Hendrycks & Gimpel, 2016:

$$
\text{GELU}(x) = x \cdot \Phi(x) \tag{8.4}
$$

其中 $\Phi$ 是标准正态分布的 CDF。常用近似:

$$
\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{2/\pi}\,(x + 0.044715 x^3)\right]\right) \tag{8.5}
$$

直觉:**"按 $x$ 自身是正值的概率加权"**。负半轴有微弱梯度,缓解死神经元;形状平滑,优化更友好。

### 8.3.3 SwiGLU(LLaMA / Mistral / DeepSeek / Qwen 默认)

Swish 激活:

$$
\text{Swish}(x) = x \cdot \sigma(\beta x), \quad \sigma(z) = \frac{1}{1+e^{-z}} \tag{8.6}
$$

当 $\beta = 1$ 时也叫 **SiLU**(Sigmoid Linear Unit)。

将 Swish 与**门控线性单元 GLU**(Dauphin et al., 2017)结合,得到 SwiGLU(Shazeer, 2020):

$$
\text{SwiGLU}(x) = \text{Swish}(W_{\text{gate}} x) \odot (W_{\text{up}} x) \tag{8.7}
$$

注意 SwiGLU **不再是单一激活**,而是带门控的两路并行结构,详见 §8.4。

### 8.3.4 三者的形状对比

| 激活 | 0 处导数 | 负半轴 | 平滑性 | 典型代表 |
| --- | --- | --- | --- | --- |
| ReLU | 不可导(尖角) | 恒 0 | C⁰ | 原版 Transformer |
| GELU | ≈ 0.5 | 有小值 | C∞ | BERT、GPT-2/3 |
| SwiGLU | 含门控 | 有小值 | C∞ | LLaMA、DeepSeek、Qwen |

经验上,SwiGLU 在相同参数量下普遍优于 GELU,几乎成为现代 LLM 默认选择。

---

## 8.4 门控线性单元(GLU)家族

### 8.4.1 GLU 的统一形式

$$
\text{GLU}_f(x) = f(W_{\text{gate}} x) \odot (W_{\text{up}} x) \tag{8.8}
$$

不同的 $f$ 衍生出不同变体:

| 变体 | $f$ | 来源 |
| --- | --- | --- |
| **GLU** | $\sigma$(sigmoid) | Dauphin 2017 |
| **ReGLU** | ReLU | Shazeer 2020 |
| **GEGLU** | GELU | Shazeer 2020 |
| **SwiGLU** | Swish/SiLU | Shazeer 2020 |

Shazeer (2020) 的消融实验显示 GEGLU / SwiGLU 在语言建模上稳定优于 ReLU / GELU,且训练稳定性更好。

### 8.4.2 SwiGLU FFN 的完整结构(LLaMA 风格)

$$
\text{FFN}_{\text{SwiGLU}}(x) = W_{\text{down}} \big[\text{Swish}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\big] \tag{8.9}
$$

三个权重矩阵(无 bias):

- $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- $W_{\text{down}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$

注意从两矩阵 (W1, W2) 变成了三矩阵 (gate, up, down)。

### 8.4.3 为什么 LLaMA 把 $d_{\text{ff}}$ 缩到约 $\frac{8}{3} d_{\text{model}}$?

为了**与 ReLU FFN 参数量持平**:

- 普通 FFN(2 矩阵):$2 \cdot d_{\text{model}} \cdot d_{\text{ff}} = 2 \cdot d \cdot 4d = 8 d^2$
- SwiGLU FFN(3 矩阵):$3 \cdot d \cdot d_{\text{ff}}$

让两者相等,得:

$$
3 d \cdot d_{\text{ff}} = 8 d^2 \;\Rightarrow\; d_{\text{ff}} = \frac{8}{3} d \approx 2.67 d \tag{8.10}
$$

LLaMA-2-7B 的实际配置:$d_{\text{model}} = 4096$,$d_{\text{ff}} = 11008 \approx \frac{8}{3} \times 4096$(并对齐到 256 的倍数)。

**记忆口诀**:**"三矩阵换两矩阵,FFN 宽度打八三折"**。

---

## 8.5 为什么 FFN 占了模型大部分参数?

### 8.5.1 单层参数量推导

记 $d = d_{\text{model}}$,$d_{\text{ff}} = 4d$(标准比例),$h$ 个 head。

**Attention 子层**(MHA,无 bias):

- $W_Q, W_K, W_V, W_O$ 各 $d \times d$
- 合计:$4d^2$

**FFN 子层**(标准两矩阵):

- $W_1: d \times 4d$,$W_2: 4d \times d$
- 合计:$8d^2$

### 8.5.2 参数比例

$$
\frac{\text{FFN}}{\text{Attention} + \text{FFN}} = \frac{8d^2}{4d^2 + 8d^2} = \frac{2}{3} \approx 67\% \tag{8.11}
$$

**FFN 大约占了 2/3**——这就是为什么"压 FFN"(MoE)比"压 Attention"(MQA/MLA)的参数收益更大的根本原因。

### 8.5.3 数字案例:LLaMA-2-7B(单层)

| 子层 | 计算 | 参数量 |
| --- | --- | --- |
| Attention(MHA) | $4 \times 4096^2$ | 67 M |
| FFN(SwiGLU,$d_{\text{ff}}=11008$) | $3 \times 4096 \times 11008$ | 135 M |
| 单层总计 | | ≈ 202 M |
| 32 层模型 | | ≈ 6.5 B(+ embedding ≈ 6.7B 全模型) |

FFN 在单层中也是 Attention 的 2 倍,与公式 8.11 吻合。

---

## 8.6 完整代码实现

### 8.6.1 标准 FFN(ReLU / GELU)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu"):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return self.w2(self.act(self.w1(x)))
```

### 8.6.2 SwiGLU FFN(LLaMA 风格)

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        # 若不指定,自动按 8/3 折算并对齐到 256
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            d_ff = ((d_ff + 255) // 256) * 256

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU = Swish(gate(x)) ⊙ up(x) → down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# === 测试 ===
if __name__ == "__main__":
    ffn = SwiGLUFFN(d_model=4096)
    x = torch.randn(2, 16, 4096)
    y = ffn(x)
    print(y.shape)             # torch.Size([2, 16, 4096])
    n = sum(p.numel() for p in ffn.parameters())
    print(f"params: {n/1e6:.1f} M")  # ≈ 135 M(与 §8.5.3 一致)
```

可直接复制运行验证。

---

## 8.7 复杂度分析

### 8.7.1 FLOPs(单 token,标准 FFN)

| 操作 | 复杂度 |
| --- | --- |
| $W_1 x$ | $O(d \cdot d_{\text{ff}})$ |
| 激活 | $O(d_{\text{ff}})$ |
| $W_2 h$ | $O(d_{\text{ff}} \cdot d)$ |
| **总计** | $\boxed{O(d \cdot d_{\text{ff}})} = O(d^2)$ |

序列长 $n$ 时为 $O(n d^2)$,**对 $n$ 是线性**——与 Attention 的 $O(n^2 d)$ 形成对比。

### 8.7.2 FFN vs Attention 谁是瓶颈?

当 $n > d$ 时,Attention 的 $n^2 d$ 超过 FFN 的 $n d^2$;反之 FFN 占主导。

- LLaMA-7B($d=4096$):序列短于 4k 时,FFN 是 FLOPs 主体;超过 4k,Attention 变主体
- 长上下文模型($n=128\text{k}$,$d=4096$):Attention 的二次项远超 FFN

**结论**: 训练长上下文时优化 Attention(Flash, MLA),训练短上下文时优化 FFN(MoE、量化)。

### 8.7.3 SwiGLU 的额外代价

SwiGLU 有 3 个大矩阵乘法(gate / up / down),比 ReLU/GELU 的 2 个多一个。但因为缩小了 $d_{\text{ff}}$(§8.4.3),**总 FLOPs 与 GELU FFN 大致持平**,质量更好。这是 SwiGLU 流行的关键工程性。

---

## 8.8 MoE:把 FFN 拆成专家

### 8.8.1 动机

公式 8.11 已经说明:FFN 占了 2/3 参数。**能不能让每个 token 只用 FFN 的一部分?** 这就是 Mixture-of-Experts(MoE)。

### 8.8.2 基本结构

将一个大 FFN 替换成 $N$ 个并行的"专家 FFN"$\{E_1, \ldots, E_N\}$ + 一个路由器(router/gate):

$$
\text{MoE}(x) = \sum_{i \in \mathcal{T}_k(x)} g_i(x) \cdot E_i(x) \tag{8.12}
$$

其中:

- $\mathcal{T}_k(x)$ 是路由器选出的 Top-$k$ 个专家(通常 $k=1$ 或 $k=2$)
- $g_i(x)$ 是路由权重(softmax 后归一)

每个 token **只激活 $k$ 个专家**,但整个模型仍能拥有 $N$ 个专家的参数容量。

### 8.8.3 关键概念

| 概念 | 含义 |
| --- | --- |
| **总参数量** | $N$ 个专家全部 = "大模型容量" |
| **激活参数量** | 单 token 实际过的参数 = $k \cdot$ 单专家大小 |
| **路由(routing)** | 路由器决定每个 token 进哪几个专家 |
| **负载均衡** | 训练时加入辅助 loss,防止"所有 token 都涌入同一个专家" |

DeepSeek-V3 的设计:**256 个专家 + 1 个共享专家,每 token Top-8 激活**。总参 671B、激活仅 37B,推理成本只与"激活参数"线性相关。详见 §23。

### 8.8.4 何时该用 MoE?

| 场景 | 选择 |
| --- | --- |
| 单 GPU 推理、参数严格受限 | Dense |
| 训练算力充足、想拉参数上限 | MoE |
| 长上下文为主、Attention 是瓶颈 | 优先优化 Attention(§24 MLA) |

---

## 8.9 与 Attention 的角色对比

| 维度 | Attention(§07) | FFN(本章) |
| --- | --- | --- |
| 输入交互 | token 间(跨位置) | token 内(逐位置) |
| 主要非线性 | softmax(温和) | ReLU/SwiGLU(强) |
| 复杂度 | $O(n^2 d)$ | $O(n d^2)$ |
| 参数占比 | ≈ 1/3 | ≈ 2/3 |
| 长上下文瓶颈 | ✅ 是 | ❌ 否 |
| 主要稀疏化方向 | KV 压缩(MLA) | 专家稀疏(MoE) |

**记忆口诀**:**Attention 横向沟通,FFN 纵向加工。前者卷长度,后者卷宽度。**

---

## 🎨 8.9.A 通俗类比合集

抽象的两层 MLP + 门控,其实就是日常生活里很熟悉的"流水线"。

### 类比一: 工厂流水线 🏭

> 一个汽车工厂,把"半成品"加工成"成品"。

- **输入 $x$** = 进入这道工序的半成品零件
- **$W_1$(上投影)** = **打散重构**: 一个零件拆成 4 倍数量的中间件,便于精细加工
- **激活函数** = **质检**: 决定每个中间件是否通过(ReLU 严格,GELU 温和,SwiGLU 还要看"配对零件")
- **$W_2$(下投影)** = **重新组装**: 把过关的中间件压回标准零件尺寸

**SwiGLU 的特别之处** = 流水线上**多了一条"门控通道"**: 主通道生产中间件,门控通道决定"每个中间件最终采用多少"——像汽车厂的"质量加权",而不是"非黑即白通过/不通过"。

### 类比二: 厨房做菜 👨‍🍳

> Attention 是"和其他厨师交流食材",FFN 是"自己埋头烹饪"。

- **$x$** = 一份食材组合
- **$W_1$** = **切配**: 把食材切成更多小块,展开成更易处理的形态
- **激活** = **加热**: 让食材发生非线性变化(没火,味道不变;有火,蛋白质变性、糖焦化…)
- **$W_2$** = **装盘**: 把所有炒好的料合成一道菜
- **SwiGLU 门控** = **盐和糖的调味勺**: 一边炒一边按比例调味,而不是炒完后再放——风味更均匀

### 类比三: 阅读理解课的"独立思考时间" 📖

> Attention 阶段类似"小组讨论",FFN 阶段类似"独立写读后感"。

- **小组讨论 (Attention)**: 大家互相听对方观点,每个 token 知道了别人在想什么
- **独立写感想 (FFN)**: 每个学生现在闭上眼睛,**只对自己刚才听到的内容**做更深的内化与加工
- **激活函数** = 学生头脑里的"筛选机制"(ReLU = 太负面的想法直接丢掉;GELU/SwiGLU = 即使负面也保留一点,允许"反直觉的思路")
- **MoE** = 学校有 256 位特长老师(数学、文学、化学…),每个学生**只去找最相关的 2 位老师**深度辅导

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🏭 工厂流水线 | **扩展—加工—压缩** | $d_{\text{ff}} = 4d$ 的"扩4压回" |
| 👨‍🍳 厨房做菜 | **非线性变换** | 激活函数为什么必不可少 |
| 📖 独立思考 | **逐 token 独立** | "Position-wise" 与 Attention 的差别 |

**记忆口诀**:
> **"先把维度撑开,再让激活筛掉,最后压回原貌。" 一句话:**FFN = 升维 × 筛选 × 降维**。**

---

## 8.10 常见疑问

### Q1: FFN 没有跨 token 交互,会不会信息不足?
**A**: FFN 本身确实只看当前 token,但它**接收的输入已经被 Attention 混合过**(来自整个序列的信息)。FFN 的工作是把混合后的特征再深加工,信息源已经全局了。

### Q2: 为什么不用更深的 FFN(比如 5 层)而是只 2 层?
**A**: Transformer 通过**层堆叠**(N 个 Block)来实现深度,不需要单层很深。每个 Block 内一层 Attention + 一层 FFN,加上残差连接,N 层堆叠后等效于 N × (注意力 + 非线性)交替。这种**横向堆叠**比单 Block 内做深更稳定。

### Q3: bias 项为什么现代 LLM 都去掉了?
**A**: 三个原因:
1. 配合 RMSNorm(§09)时,bias 对输出分布几乎无影响;
2. 减少参数(每层省 $d_{\text{model}} + d_{\text{ff}}$);
3. 实证上无 bias 的训练稳定性更好(PaLM、LLaMA 都已验证)。

### Q4: SwiGLU 中的 gate 和 up 可以共享权重吗?
**A**: 数学上可以,但**等价于退化成 Swish 激活**(失去门控能力)。Shazeer 的消融显示双矩阵显著优于单矩阵,值得多花的参数。

### Q5: MoE 推理时,token 路由是固定的吗?
**A**: 不是。每个 token 经过 router 实时计算 Top-k 专家,因此**同一 batch 的不同 token 会进不同专家**。这给推理引擎带来负载不均衡问题,需要专门的调度(详见 §23 DeepSeekMoE 与 §21 推理优化)。

---

## 8.11 本章小结

### 核心公式

**标准 FFN**:

$$
\text{FFN}(x) = W_2 \,\sigma(W_1 x) \quad (d_{\text{ff}} = 4 d_{\text{model}})
$$

**SwiGLU FFN(现代默认)**:

$$
\text{FFN}_{\text{SwiGLU}}(x) = W_{\text{down}}\big[\text{Swish}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\big] \quad (d_{\text{ff}} \approx \tfrac{8}{3} d_{\text{model}})
$$

**MoE**:

$$
\text{MoE}(x) = \sum_{i \in \text{Top-}k} g_i(x) E_i(x)
$$

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **角色** | 每 token 独立深加工,Attention 的互补 |
| **结构** | 升维 → 激活 → 降维 |
| **激活演进** | ReLU → GELU → SwiGLU |
| **参数主体** | FFN ≈ 模型 2/3 参数 |
| **复杂度** | $O(n d^2)$,对 $n$ 线性 |
| **稀疏化** | MoE = 大容量、低激活 |

### 一句话总结

> **FFN 是 Transformer Block 的非线性主力,负责把 Attention 混合好的特征"升维—激活—降维"做深加工;它占模型 2/3 参数,所以稀疏化(MoE)收益最大;现代 LLM 普遍采用 SwiGLU,以多一个矩阵的代价换取更优的训练稳定性和效果。**

---

## 8.12 延伸阅读

### 必读论文

1. Vaswani et al., 2017. *"Attention is All You Need"*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)(原版 FFN)
2. Shazeer, 2020. *"GLU Variants Improve Transformer"*. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)(SwiGLU 来源)
3. Hendrycks & Gimpel, 2016. *"Gaussian Error Linear Units (GELUs)"*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)

### 进阶论文

4. Dauphin et al., 2017. *"Language Modeling with Gated Convolutional Networks"*.(GLU 起源)
5. Fedus et al., 2022. *"Switch Transformer"*. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)(MoE 工业级实现)
6. DeepSeek-AI, 2024. *"DeepSeek-V3 Technical Report"*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)(细粒度专家 + 共享专家)
7. Touvron et al., 2023. *"LLaMA: Open and Efficient Foundation Language Models"*. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

### 推荐资源

- Lilian Weng: *"The Transformer Family v2"* 博客
- HuggingFace Blog: *"Mixture of Experts Explained"*

---

## 8.13 实战练习

### 练习 1 (★)

修改 §8.6.1 的 `StandardFFN`,加入对 `silu` 激活的支持,并对比同 $d_{\text{ff}}$ 下 ReLU / GELU / SiLU 三种激活在 1k 步小数据集上的 loss 曲线。

### 练习 2 (★★)

实现一个不带 bias 的 SwiGLU,使其参数量与标准 GELU FFN($d_{\text{ff}}=4d$)**严格相等**——亲手验证 §8.4.3 的"8/3 缩放"。

### 练习 3 (★★)

给定 $d_{\text{model}}=4096$、$L=32$ 层、$d_{\text{ff}}=11008$、词表 32000,推算 LLaMA-2-7B 的总参数量(embedding + L × (Attention + FFN) + LM head),并对比实际公布数字 6.738B。

### 练习 4 (★★★)

实现一个最小可运行的 **Top-2 MoE FFN**(4 个专家),要求:
1. router 输出 Top-2 专家与权重;
2. 加入"load-balancing auxiliary loss";
3. 在玩具数据上验证 routing 不退化为单一专家。

### 练习 5 (★★★)

理论推导:在 $d_{\text{ff}} = \alpha d_{\text{model}}$ 时,SwiGLU FFN 与 ReLU FFN 在 **FLOPs 持平**的条件下 $\alpha$ 应该是多少?($\alpha = 8/3$ 是参数持平,FLOPs 持平稍有差异。)

---

## 章节交叉引用

- 前置: [§01 线性代数](../Part_I_数学与基础/01_线性代数与矩阵运算.md), [§07 Self-Attention](07_Self_Attention机制.md)
- 后续: [§09 归一化与残差](09_归一化与残差.md), [§10 Transformer Block 整体](10_Transformer_Block整体.md), [§23 DeepSeekMoE](../Part_V_DeepSeek专题/23_DeepSeekMoE.md)
- 相关: [§17 Scaling Law](../Part_III_训练原理/17_Scaling_Law.md), [§21 推理优化](../Part_IV_推理与部署/21_推理优化.md)

---

*下一章: §09 归一化与残差(LayerNorm / RMSNorm / Pre-Norm vs Post-Norm)*
