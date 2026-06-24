# 25 DeepSeekMoE(细粒度专家 + 共享专家 + 负载均衡 + Auxiliary-Loss-Free)

> **章节定位**: Part V 第 3 章,把 §08.8 的 MoE 引论扩到工程级
> **预计阅读时间**: 80-100 分钟
> **难度**: ★★★★☆
> **前置知识**: §08 FFN/MoE 引论、§14 EP 分布式、§16 KL 与辅助 loss
> **后续依赖**: §26 V3 工程、§27 V3 vs LLaMA-3

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [25.1 朴素 MoE 的三大痛](#251-朴素-moe-的三大痛)
- [25.2 DeepSeekMoE 三件套](#252-deepseekmoe-三件套)
- [25.3 细粒度专家(Fine-Grained Expert Segmentation)](#253-细粒度专家fine-grained-expert-segmentation)
- [25.4 共享专家(Shared Expert Isolation)](#254-共享专家shared-expert-isolation)
- [25.5 路由器:Top-k 与归一化](#255-路由器top-k-与归一化)
- [25.6 负载均衡:辅助 loss(V2 风格)](#256-负载均衡辅助-lossv2-风格)
- [25.7 Auxiliary-Loss-Free Balancing(V3 创新)](#257-auxiliary-loss-free-balancingv3-创新)
- [25.8 Device-Limited Routing & 通信约束](#258-device-limited-routing--通信约束)
- [25.9 推理时的专家路由](#259-推理时的专家路由)
- [25.10 完整代码实现:可运行 mini DeepSeekMoE](#2510-完整代码实现可运行-mini-deepseekmoe)
- [25.11 V2 / V3 实际配置](#2511-v2--v3-实际配置)
- [🎨 25.11.A 通俗类比合集](#-2511a-通俗类比合集)
- [25.12 常见疑问](#2512-常见疑问)
- [25.13 本章小结](#2513-本章小结)
- [25.14 延伸阅读](#2514-延伸阅读)
- [25.15 实战练习](#2515-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

朴素 MoE(每层 8-16 专家,每 token Top-2)在 LLM 上长期不稳:专家分化不足、负载偏斜、辅助 loss 与主 loss 冲突。**DeepSeekMoE** 用三件套把 MoE 推到生产级:

1. **细粒度专家**:把 1 个 4d 大 FFN 切成 N 个小专家,**总参数不变,激活更稀疏更专业**
2. **共享专家**:1-2 个永远激活的专家承担"通用知识",让专精专家真正专精
3. **负载均衡新范式**:V2 用细致的辅助 loss;**V3 直接去掉辅助 loss**,用 bias 微调路由器实现"无损均衡"

读完本章你能解释 DeepSeek-V3 的 "256 routed expert + 1 shared + Top-8" 为何是经过精心校准的配置,以及它在训练 / 推理两端如何与 EP(§14.8)、DualPipe(§26)配合。

---

## 学习目标

完成本章后,你能够:

- ✅ 解释"细粒度专家"如何在**参数预算固定**下提升表达力 (§25.3)
- ✅ 写出共享专家 + Top-k routed 的完整 MoE 公式 (公式 25.3)
- ✅ 推导辅助 loss(V2)的形式与作用 (§25.6)
- ✅ 解释 V3 的 auxiliary-loss-free 路由如何工作 (§25.7)
- ✅ 计算 V3 配置的"激活参数 vs 总参数" (§25.11)

---

## 25.1 朴素 MoE 的三大痛

回顾 §08.8:朴素 MoE FFN 用 N 个大专家(每个大小 ≈ dense FFN),Top-k 路由。它的痛点是:

### 痛 1:专家分化不足

8-16 个大专家中,模型很容易让多数 token 进同一两个专家(强者更强),其他专家长期闲置 → 等效参数远小于名义参数。

### 痛 2:负载严重偏斜

实际训练里,前 5% 专家可能承接 80% token,GPU 间负载差几个数量级 → 慢的拖累整体。

### 痛 3:辅助 loss 与主 loss 冲突

为了均衡负载,通常加 KL 到均匀 / 方差正则等辅助 loss。代价:**辅助 loss 直接梯度回主模型,污染语言建模信号**,质量下降。

DeepSeekMoE 的三个改进对症下药。

---

## 25.2 DeepSeekMoE 三件套

| 改进 | 解决什么 | 原理 |
| --- | --- | --- |
| **细粒度专家(Fine-grained)** | 痛 1 | 同参数下专家更多更小,分工更清晰 |
| **共享专家(Shared)** | 痛 1 + 通用知识冗余 | 1-2 个永远激活专家承担通用部分 |
| **Auxiliary-Loss-Free 路由(V3)** | 痛 2 + 痛 3 | 用 bias 修正而非梯度污染 |

后面三节分别展开。

---

## 25.3 细粒度专家(Fine-Grained Expert Segmentation)

### 25.3.1 直觉

把 1 个隐藏维 $d_{\text{ff}}$ 的专家切成 $m$ 份,每份维度 $d_{\text{ff}}/m$。**总参数不变**,但:

- 专家数 $N$ 翻 $m$ 倍($N \to mN$)
- 每 token 激活专家数 $k$ 也翻 $m$ 倍($k \to mk$)
- 总激活 FLOPs **不变**($k \cdot d_{\text{ff}}$)

但路由组合数变大:$\binom{N}{k}$ 远大于原值 → **表达组合空间几何级增加**。

### 25.3.2 公式形式

定义 $N$ 个 routed 专家 $\{E_i\}_{i=1}^N$,每个是小 SwiGLU FFN(隐藏维 $d_e \ll d_{\text{ff}}$)。Top-k 选择 + softmax 权重:

$$
\text{MoE}(x) = \sum_{i \in \mathcal{T}_k(x)} g_i(x) \cdot E_i(x) \tag{25.1}
$$

其中

$$
g_i(x) = \frac{\exp(z_i)}{\sum_{j \in \mathcal{T}_k} \exp(z_j)}, \quad z = W_{\text{router}}\, x
$$

### 25.3.3 V2 / V3 的具体数字

| 模型 | 总专家 $N$ | 每 token 激活 $k$ | 单专家隐藏维 $d_e$ |
| --- | --- | --- | --- |
| Mixtral 8×7B | 8 | 2 | 14336(大) |
| DeepSeek-V2 | 160(2 共享 + 158 routed) | 6 routed | 1408(小) |
| **DeepSeek-V3** | **257(1 共享 + 256 routed)** | **8 routed** | **2048** |

V3 的专家是 Mixtral 的 ~10×(密度更高),但每个体量小一个数量级 → 同样的激活算力下,**组合可能性多几个量级**。

### 25.3.4 实证

DeepSeek-V2 论文 ablation:相同激活参数下,细粒度专家(160)比粗粒度(16)在 MMLU 上提升 ~2-3 个点,且训练曲线更平滑。

---

## 25.4 共享专家(Shared Expert Isolation)

### 25.4.1 动机

观察:专家间普遍存在"通用知识"(比如基础语法、高频实体)冗余。每个专家都得学一份相同的"地基" → 浪费容量。

### 25.4.2 解法

固定 1-2 个 **shared expert**,**对所有 token 永远激活**。它们承担通用知识,routed 专家不再重复学。

完整 MoE 输出:

$$
\text{MoE}(x) = \underbrace{\sum_{s=1}^{N_s} E^{\text{shared}}_s(x)}_{\text{共享部分}} + \underbrace{\sum_{i \in \mathcal{T}_k(x)} g_i(x) \cdot E_i(x)}_{\text{Top-k 路由}} \tag{25.2}
$$

(注:V3 的实现把 shared 也乘 sigmoid 或 ones,各家略有差异;核心是"无条件激活")。

### 25.4.3 收益

- routed 专家"被迫"分化(共享专家替它分担通用),专精程度上升
- 整体激活参数略多一点(shared 是固定开销),但质量提升明显

V3 配置:1 个 shared + 8 routed,所以**每 token 实际激活 9 个专家**(标的"Top-8" 指 routed 部分)。

---

## 25.5 路由器:Top-k 与归一化

### 25.5.1 Router 是个小 Linear

$$
z = W_{\text{router}}\, x, \quad z \in \mathbb{R}^N \tag{25.3}
$$

通常 router 是 $d_{\text{model}} \to N$ 的 nn.Linear,无 bias。**全模型参数中 router 体量可忽略**。

### 25.5.2 Top-k 选择

按 $z_i$ 取最大 $k$ 个,对应索引集合 $\mathcal{T}_k$。

### 25.5.3 Softmax 归一化

只对 Top-k 内部做 softmax,权重之和 = 1:

$$
g_i = \frac{\exp(z_i)}{\sum_{j \in \mathcal{T}_k} \exp(z_j)}, \quad i \in \mathcal{T}_k
$$

### 25.5.4 与 sigmoid 路由(V3 起)

V3 论文报告也试验过 **sigmoid 路由**(每专家独立 0-1 权重),配合 group-limited routing,稳定性更好。具体见 V3 §2.2。

### 25.5.5 Routing 的不可微性

$\arg\max$ / Top-k 不可微 → router 只通过被选中专家的梯度回传(类似 hard attention)。这是 MoE 训练不稳的根源之一,需要负载均衡机制支撑。

---

## 25.6 负载均衡:辅助 loss(V2 风格)

### 25.6.1 V2 的 loss 套件

V2 引入三项辅助 loss:

1. **Expert balance**:让每个专家被选中的频率接近均匀
2. **Device balance**:让每张卡承接的 token 总数接近均匀(EP 时关键)
3. **Communication balance**:控制跨节点 All-to-All 量

形式举例(expert balance):

$$
\mathcal{L}_{\text{bal}} = N \cdot \sum_{i=1}^{N} f_i \cdot p_i
$$

其中 $f_i$ 是专家 $i$ 在当前 batch 被选中的频率,$p_i$ 是 router 给专家 $i$ 的平均概率。该 loss 鼓励 "frequency × probability" 均匀分布。

### 25.6.2 辅助 loss 的副作用

辅助 loss 与主 loss 共享同一份梯度通道 → 优化器要同时迎合两边,**质量上常见 0.2-0.5 PPL 损失**。

V2 通过精心调权(辅助 loss 系数 0.001-0.01)缓解。

---

## 25.7 Auxiliary-Loss-Free Balancing(V3 创新)

### 25.7.1 核心想法

> 不通过梯度去推 router 均衡,而是**给 router 输出加一个"动态 bias"** $b_i$,实时纠正负载偏斜。

$$
z'_i = z_i + b_i, \quad \text{Top-k 用 } z' \text{ 选,但权重 } g \text{ 仍用 } z \text{ 算} \tag{25.4}
$$

### 25.7.2 bias 怎么更新

每 step 监控每个专家被选中的负载 $f_i$:

- $f_i$ 偏高(超载) → 把 $b_i$ 微调一个小负值($-\gamma$),下 step 让它"略不容易"被选
- $f_i$ 偏低(欠载) → $b_i$ 加 $+\gamma$

$\gamma$ 通常很小(1e-3 量级)。bias 是**不进梯度图的纯统计调节器**,不污染主 loss。

### 25.7.3 为什么 Top-k 用 $z'$ 但 $g$ 用 $z$?

如果权重 $g$ 也用带 bias 的 $z'$,那 bias 会扭曲专家输出加权 → 实际行为偏离 router 学习目标。**只让 bias 影响"路由选择",不影响"组合权重"**——这是设计的精妙处。

### 25.7.4 V3 的实证

DeepSeek-V3 报告表明:

- 与同等带辅助 loss 的设置相比,无辅助 loss 路由的最终 PPL 略低
- 负载均衡度并不退化
- 工程简单(少了 loss 项及其系数调参)

**意义**:把"负载均衡"从"模型层学习目标"剥离成"训练时的纯统计反馈",让主 loss 干净。这是 V3 在 MoE 范式上最优雅的贡献之一。

---

## 25.8 Device-Limited Routing & 通信约束

### 25.8.1 EP 下的痛

§14.8:MoE 用 Expert Parallel,把 N 个专家分布到多卡,token 通过 All-to-All 路由到对应卡。

如果 token A 的 Top-8 专家分布在 8 张卡上 → 8 次跨卡通信。当 N=256 时,通信量爆炸。

### 25.8.2 Device-Limited Routing

V3 把 routed 专家分成 $G$ 个**专家组**(group),每个 token **至多激活 $M$ 个组**($M < G$),保证一个 token 的 Top-k 专家**只散落在 M 张卡内**。

例:256 专家分成 8 组(每组 32 个),Top-M=4 组 → token 通信范围 4 张卡,而非 8。

### 25.8.3 Group-Limited Top-k

实现细节:先按组内最高分对组排序,选 Top-M 组,再在这 M 组内取 Top-k 专家。

代价:略损路由灵活性;收益:通信量大降。配合 DualPipe(§26)的通信-计算重叠,可达到几乎"通信免费"。

---

## 25.9 推理时的专家路由

### 25.9.1 路由相同,EP 配置可不同

训练时 EP=64(每卡 4 个专家);推理时可改 EP=8(每卡 32 个专家)或本地全部加载。

### 25.9.2 推理性能挑战

不同 batch 不同 token 路由结果不一,**单 step 算力 / 显存高度不规则**。需要:

- 高效 All-to-All(可能要求 RDMA / NVLink)
- 拓扑感知:路由热点专家放高带宽位置
- 动态调度:专家算力打表均衡

V3 推理在 vLLM / SGLang 上已稳定,主要靠社区 + DeepSeek 自有 kernel。

### 25.9.3 量化与 MoE

权重量化(AWQ INT4)同样适用 MoE 专家,但要注意**每专家独立校准**(同一校准数据可能命中不到所有专家)。社区做法:多轮校准 + 专家级 scale。

---

## 25.10 完整代码实现:可运行 mini DeepSeekMoE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallExpert(nn.Module):
    """小 SwiGLU FFN。"""
    def __init__(self, d_model, d_e):
        super().__init__()
        self.gate = nn.Linear(d_model, d_e, bias=False)
        self.up   = nn.Linear(d_model, d_e, bias=False)
        self.down = nn.Linear(d_e, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DeepSeekMoE(nn.Module):
    def __init__(self, d_model: int, n_routed: int, n_shared: int,
                 d_e: int, top_k: int):
        super().__init__()
        self.n_routed, self.n_shared, self.top_k = n_routed, n_shared, top_k
        self.shared = nn.ModuleList([SmallExpert(d_model, d_e) for _ in range(n_shared)])
        self.routed = nn.ModuleList([SmallExpert(d_model, d_e) for _ in range(n_routed)])
        self.router = nn.Linear(d_model, n_routed, bias=False)
        # auxiliary-loss-free balancing 的 bias(不参与梯度)
        self.register_buffer("bias", torch.zeros(n_routed))
        # 训练时统计每专家被选中次数,用于更新 bias
        self.register_buffer("load_count", torch.zeros(n_routed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) — 这里展开成 token 维度处理
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)                       # (N, d)
        N_tok = x_flat.size(0)

        # 1. shared 专家:对所有 token 永远激活
        out = sum(e(x_flat) for e in self.shared)

        # 2. routed:用带 bias 的分数选 Top-k,但 g 用无 bias 分数算
        z = self.router(x_flat)                            # (N, n_routed)
        z_biased = z + self.bias
        top_vals, top_idx = z_biased.topk(self.top_k, dim=-1)   # (N, k)
        # gate 权重用原始 z 的对应位置
        gate_logits = z.gather(-1, top_idx)
        gate = F.softmax(gate_logits, dim=-1)              # (N, k)

        # 3. 累加 routed 专家输出(教学版,遍历 expert;生产用稀疏 gather)
        for slot in range(self.top_k):
            for e in range(self.n_routed):
                mask = (top_idx[:, slot] == e)             # 哪些 token 这个 slot 走 e
                if mask.any():
                    out_e = self.routed[e](x_flat[mask])
                    out[mask] += gate[mask, slot:slot+1] * out_e

        # 4. 更新负载统计(训练时)
        if self.training:
            with torch.no_grad():
                cnt = torch.bincount(top_idx.flatten(), minlength=self.n_routed).float()
                self.load_count += cnt

        return out.view(B, T, d)

    @torch.no_grad()
    def update_bias(self, gamma: float = 1e-3):
        """周期性调用:按当前 load_count 更新 bias 实现均衡。"""
        target = self.load_count.mean()
        # 超载 → bias 减;欠载 → bias 加
        self.bias -= gamma * (self.load_count - target).sign()
        self.load_count.zero_()
```

教学版重点:**bias 不进梯度图(register_buffer)、Top-k 用 bias 修正、gate 权重仍用原始 logits**。生产代码(DeepEP / Megatron-Core / vLLM)用稀疏 gather + EP All-to-All,效率高得多。

---

## 25.11 V2 / V3 实际配置

### 25.11.1 V2 配置

| 项 | 值 |
| --- | --- |
| Total parameters | 236 B |
| Activated parameters | 21 B |
| Layers | 60 |
| 第 1 层 | dense FFN(不 MoE) |
| 第 2-60 层 | MoE |
| 每层 routed experts | 160 |
| 每层 shared experts | 2 |
| Top-k(routed) | 6 |
| 单 routed expert 隐藏维 | 1408 |
| 负载均衡 | 辅助 loss(expert + device + comm) |

### 25.11.2 V3 配置

| 项 | 值 |
| --- | --- |
| Total parameters | 671 B |
| Activated parameters | 37 B |
| Layers | 61 |
| 第 1 / 2 / 3 层 | dense FFN |
| 第 4-61 层 | MoE |
| 每层 routed experts | 256 |
| 每层 shared experts | 1 |
| Top-k(routed) | 8 |
| 单 routed expert 隐藏维 | 2048 |
| 负载均衡 | **Auxiliary-Loss-Free**(bias) |
| Device-limited routing | $M = 4$(每 token 至多 4 卡) |

### 25.11.3 激活参数核算

V3 每 token 激活:

- Attention(MLA):全模型 ~5B
- Shared expert:1 个
- Routed expert:8 个
- Embedding / Norm / LM head: 少量

合计 ≈ 37 B(论文给出)。占总参数 671 B 的 **5.5%**。

这就是为什么 V3 的训练 / 推理算力按 dense ~37 B 算,但模型容量按 dense ~150-200 B(有效参数 $\sqrt{P_t \cdot P_a}$,§17.6)算。

---

## 🎨 25.11.A 通俗类比合集

### 类比一: 大型综合医院 🏥

> 朴素 MoE = 8 个全科主任;DeepSeekMoE = 256 个专科 + 1 个全科会诊。

- **细粒度专家** = 不养 8 个超级全科主任,而是养 256 位**心脏 / 神经 / 眼科**等小专科,每病例分流到对症的 8 位
- **共享专家** = 院里始终值班的全科会诊医生,所有病例都先经手做基础检查 → 专科医生不必再做"基础体检"那些事
- **辅助 loss 负载均衡** = 院长开会强行给每个科室分配同等病人,科室间不情愿
- **Auxiliary-Loss-Free** = 院方实时看排队长度,**默默调整分诊指引(bias)**,让流量自然平衡,医生不知道
- **Device-limited routing** = 一个病例只在同一楼层内的专科间转,不去对面楼避免坐电梯(节省通信)

### 类比二: 巨大邮局分拣 ✉️

> 朴素 MoE = 16 个全科分拣员;DeepSeekMoE = 256 个邮编专区分拣员。

- **细粒度** = 邮编越细分,差错越少;每封信只去 8 个最相关分区
- **共享专家** = 每封信先经过一个"通用预处理"(称重 / 扫码),再分到专区
- **辅助 loss** = 邮政局长强迫每分区当天工作量必须相等,但实际邮件分布不均,造成冲突
- **Auxiliary-Loss-Free** = 通过"队列长度看板"动态调整入站规则,不需要硬性平摊
- **Device-limited routing** = 一封信的分拣不许跨城市,只在同一邮局内转手

### 类比三: 大型咨询公司项目分配 🏢

> 朴素 MoE = 16 位全能合伙人;DeepSeekMoE = 256 位垂直行业咨询师 + 1 个老板。

- **共享专家** = 老板 (Managing Partner) 永远参与每个项目,提供战略视角
- **细粒度** = 256 位垂直专家(零售/医疗/金融/科技…),每个项目派 8 位
- **细粒度让分工自然涌现** = 不需要预告"你专心做零售",而是让 router 把"零售类项目"自然分给某些人
- **负载均衡** = HR 看每人本周项目数,动态调整 "投标资格"(bias),不通过强制 KPI
- **Device-limited** = 一个项目的人员只在一个办公室内组队,避免飞来飞去

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🏥 医院 | **专科分工 + 全科兜底** | 细粒度 + 共享专家 |
| ✉️ 邮局 | **流量均衡** | 辅助 loss vs bias-based |
| 🏢 咨询 | **组合多样性** | Top-k 路由空间 |

**记忆口诀**:
> **"专家细分 + 共享兜底 + bias 平衡;V3 = 256 细专家 + 1 共享 + Top-8 + 无辅助 loss。"**

---

## 25.12 常见疑问

### Q1: 共享专家是不是浪费?它和"dense FFN"有区别吗?
**A**: 共享 + routed 总参数仍 << 等效 dense。共享专家承担"通用部分",routed 承担"专精部分",两者协同。直接 dense 反而损失稀疏带来的容量优势。

### Q2: Top-8 是不是太多?
**A**: 看你怎么看:
- 与 Mixtral 的 Top-2 比 → 多
- 与"V3 一共有 256 个专家"比 → 仅 3% → 仍非常稀疏

V3 论文显示 Top-8 是质量/通信的甜点(更大 Top 提升边际下降,通信开销线性上升)。

### Q3: Auxiliary-Loss-Free 真的更好吗?
**A**: V3 ablation 显示:同等设置下少 0.05-0.1 PPL,且负载更均衡。简单且有效——这种"做减法"的创新最珍贵。

### Q4: 加新专家可以热插拔吗(MoE 模型扩展性)?
**A**: 理论可以(初始化新专家 + 微调 router);但路由器与现有专家紧耦合,工程上比想象的难。社区少见纯热插拔成功案例。

### Q5: MoE 模型推理时是否所有专家都要装显存?
**A**: 是(理论上)。每 token 路由可能命中任意专家。Offloading 到 CPU 可行但慢。这就是为什么 V3 推理要求多卡 / 大显存。

---

## 25.13 本章小结

### 核心公式

**DeepSeekMoE FFN**:

$$
\text{MoE}(x) = \sum_{s=1}^{N_s} E^{\text{shared}}_s(x) + \sum_{i \in \mathcal{T}_k(x)} g_i(x) \cdot E_i(x)
$$

**Auxiliary-Loss-Free Top-k 选择**:

$$
\mathcal{T}_k = \text{TopK}_i(z_i + b_i), \quad g_i = \text{softmax}(z_i)_{i \in \mathcal{T}_k}
$$

**bias 动态更新**:

$$
b_i \leftarrow b_i - \gamma \cdot \text{sign}(f_i - \bar f)
$$

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **细粒度专家** | 同参数下提高组合空间 |
| **共享专家** | 通用知识分担,让路由专家专精 |
| **Auxiliary-Loss-Free** | bias 实时纠偏,不污染主 loss |
| **Device-limited routing** | 控制 All-to-All 通信范围 |
| **V3 配置** | 256 routed + 1 shared + Top-8 + bias |

### 一句话总结

> **DeepSeekMoE 把 MoE 工程化的三步走:细粒度专家让分工自然涌现、共享专家承担通用知识、auxiliary-loss-free 路由让负载均衡不再污染主 loss;再配 device-limited routing 控制通信范围——使 V3 用 37B 激活参数撬动 671B 总容量,在 EP+DualPipe 系统(§14+§26)上落地。**

---

## 25.14 延伸阅读

### 必读论文

1. Dai et al., 2024. *"DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"*. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
2. DeepSeek-AI, 2024. *"DeepSeek-V2"*(完整 MoE 工程描述). [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
3. DeepSeek-AI, 2024. *"DeepSeek-V3 Technical Report"* §2.2(auxiliary-loss-free). [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

### 进阶论文

4. Fedus et al., 2022. *"Switch Transformer"*(MoE 工业化原始)[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
5. Lepikhin et al., 2020. *"GShard"*(EP 与负载均衡早期)
6. Mixtral 8×7B / 8×22B 技术报告

### 推荐资源

- HuggingFace blog: *"Mixture of Experts Explained"*
- DeepSeek 官方推理 kernel DeepEP / FlashMoE
- Megatron-Core MoE 实现

---

## 25.15 实战练习

### 练习 1 (★)

读 DeepSeekMoE 论文 Figure 2(细粒度专家与负载分布),用一段话总结观察。

### 练习 2 (★★)

把 §25.10 的 `DeepSeekMoE` 替换 §10.6 GPTMini 的 FFN,在 tiny shakespeare 上训练 1000 步,记录:
- 每个专家被选中次数
- bias 收敛曲线
- loss 曲线 vs 同等参数 dense FFN

### 练习 3 (★★)

把 §25.10 中 `Auxiliary-Loss-Free` 改成"加一个 KL 到均匀的辅助 loss",对比两种方案的负载方差与最终 PPL。

### 练习 4 (★★★)

实现 Device-limited routing:把 64 个专家分成 8 组,top_k=4,token 至多激活 2 组;对比无 group-limit 的版本在跨设备通信量上的差异(用模拟数)。

### 练习 5 (★★★)

读 V3 §2.2 与 DeepSeekMoE 原论文 §3,梳理 V2 → V3 在 MoE 上的所有改动,讨论哪一项对 V3 671B 的训练成本下降贡献最大。

---

## 章节交叉引用

- 前置: [§08 FFN 与 MoE 引论](../Part_II_Transformer架构/08_FFN与激活函数.md), [§14 EP 分布式](../Part_III_训练原理/14_分布式训练.md), [§16 辅助 loss](../Part_III_训练原理/16_偏好对齐.md)
- 后续: [§26 V3 训练工程](26_V3_训练曲线与工程.md), [§27 V3 vs LLaMA-3](27_V3_vs_LLaMA3.md)
- 相关: [§17 Scaling Law(MoE 部分)](../Part_III_训练原理/17_Scaling_Law.md), [§22 部署](../Part_IV_推理与部署/22_本地部署实战.md)

---

*下一章: §26 V3 训练曲线与工程*
