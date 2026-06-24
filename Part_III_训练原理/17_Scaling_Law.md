# 17 Scaling Law(Kaplan / Chinchilla / 现代演进 / Inference-Aware)

> **章节定位**: Part III 收尾,把"训练"决策抽象成可预测的工程公式
> **预计阅读时间**: 60-90 分钟
> **难度**: ★★★★☆
> **前置知识**: §10 总参数公式、§11 数据、§13 训练循环、§04 优化
> **后续依赖**: §22 部署经济学、§27 DeepSeek-V3 系统设计

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [17.1 现象:loss 是 P / D / C 的幂律](#171-现象loss-是-p--d--c-的幂律)
- [17.2 Kaplan 等 (2020):第一代 scaling law](#172-kaplan-等-2020第一代-scaling-law)
- [17.3 Chinchilla (2022):重新校准](#173-chinchilla-2022重新校准)
- [17.4 Compute-Optimal vs Inference-Optimal](#174-compute-optimal-vs-inference-optimal)
- [17.5 Emergent Abilities(涌现)的争议](#175-emergent-abilities涌现的争议)
- [17.6 MoE 与稀疏模型的 scaling](#176-moe-与稀疏模型的-scaling)
- [17.7 数据约束 / 重复 token 的 scaling](#177-数据约束--重复-token-的-scaling)
- [17.8 Test-time Scaling(R1 / o1 路线)](#178-test-time-scalingr1--o1-路线)
- [17.9 工程实用:给你一笔预算,怎么花?](#179-工程实用给你一笔预算怎么花)
- [17.10 完整代码:扫小模型估算 scaling](#1710-完整代码扫小模型估算-scaling)
- [🎨 17.10.A 通俗类比合集](#-1710a-通俗类比合集)
- [17.11 常见疑问](#1711-常见疑问)
- [17.12 本章小结](#1712-本章小结)
- [17.13 延伸阅读](#1713-延伸阅读)
- [17.14 实战练习](#1714-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

LLM 不是"调试出来"的,而是"按 scaling law **算**出来的"。本章梳理这条贯穿整个 LLM 时代的工程主线:

- **现象**:测试 loss 与参数 $P$、数据 $D$、算力 $C$ 呈幂律关系
- **Kaplan 2020 → Chinchilla 2022**:从"大模型小数据"到"$D \approx 20 P$"
- **Inference-aware scaling**:把推理成本纳入"compute-optimal",催生了 7B / 13B 这种"过训练小模型"
- **MoE scaling**:激活参数 vs 总参数的 scaling 是不同曲线
- **Test-time scaling**(o1 / R1):reasoning 长度作为新的"compute"维度
- 给个预算,怎么把它最优地分给"模型大小 / 训练 token / 推理 budget"

读完你能给老板算出"花 $5M 训什么尺寸最值"。

---

## 学习目标

完成本章后,你能够:

- ✅ 写出 Chinchilla 的最优比例 $D^*/P^*$ (公式 17.4)
- ✅ 解释 LLaMA-2 / LLaMA-3 为何"过训练"小模型 (§17.4)
- ✅ 区分 emergent ability 与 metric artifact(§17.5)
- ✅ 解释 MoE 的有效参数与稀疏 scaling (§17.6)
- ✅ 给定 $C$,推算最优 $(P, D)$ (§17.9)

---

## 17.1 现象:loss 是 P / D / C 的幂律

固定 Transformer 架构、用 NTP loss,大量实验显示 cross-entropy 与三个变量近似服从幂律:

$$
L(P) \approx A \cdot P^{-\alpha} + L_\infty, \quad L(D) \approx B \cdot D^{-\beta} + L_\infty \tag{17.1}
$$

$L_\infty$ 是文本熵的不可约下限(语言本身的不确定性)。$\alpha, \beta$ 经验上 ~ 0.07-0.1。

**Scaling Law** 不是定理,而是**经验定律**:它在多个数量级上稳定成立,使我们能用小模型外推大模型行为。

---

## 17.2 Kaplan 等 (2020):第一代 scaling law

### 17.2.1 主要结论

Kaplan et al., OpenAI 2020 用 GPT-2 系列扫参得到:

$$
L(N, D, C) \approx L_\infty + \left[\left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}\right] \tag{17.2}
$$

其中 $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$。

**关键推论**:给定算力 $C$,**模型尺寸 $N$ 应远比数据 $D$ 增长更快**(Kaplan 建议 $N \propto C^{0.73}$,$D \propto C^{0.27}$)。

### 17.2.2 影响

GPT-3 175B 用仅 300B token 训练 → 严格按 Kaplan 配方。当时这种"大模型小数据"被视为正统。

---

## 17.3 Chinchilla (2022):重新校准

### 17.3.1 重新做实验

Hoffmann et al.(DeepMind 2022)用更系统的 grid search(70M-16B 模型,5B-500B token,共 400+ 训练),发现 Kaplan 估算的 $\alpha_N$ 偏低。

### 17.3.2 新结论:模型 vs 数据应等速增长

$$
L(N, D) \approx L_\infty + A N^{-\alpha} + B D^{-\beta}, \quad \alpha \approx \beta \approx 0.34 \tag{17.3}
$$

固定算力 $C = 6 N D$,最优分配:

$$
\boxed{\;D^* \approx 20 \cdot N^*\;} \tag{17.4}
$$

即:**每 1 个参数,大约对应 20 个训练 token**。

### 17.3.3 Chinchilla 70B 的实证

DeepMind 用上述配方训了 Chinchilla 70B / 1.4T token,在多数 benchmark 上**击败了 Gopher 280B / 300B token**——**用 1/4 参数 + 5× token 数,效果更好**。

### 17.3.4 影响

整个行业重新校准:LLaMA-1 7B 用 1T、13B/33B 用 1.4T、65B 用 1.4T 都接近或超过 Chinchilla 推荐。这成为 2023+ 的训练默认 baseline。

---

## 17.4 Compute-Optimal vs Inference-Optimal

### 17.4.1 Chinchilla 只考虑了训练成本

但模型一旦训出,**会被推理很多次**。推理成本与 $P$ 线性相关 → 模型越小,长期推理越省。

### 17.4.2 Inference-aware Scaling

把推理需求计入总成本:

$$
C_{\text{total}} = 6 P D \;+\; 2 P D_{\text{infer}}
$$

其中 $D_{\text{infer}}$ 是预计推理总 token 数。如果 $D_{\text{infer}} \gg D$,**最优 $P$ 应小于 Chinchilla**——把多余 budget 用来训更多 token。

### 17.4.3 LLaMA 系列的实证

- LLaMA-2 7B 用 2T token,Chinchilla 推荐才 ~140B → **过训练 14×**
- LLaMA-3 8B 用 15T token,Chinchilla 推荐 ~160B → **过训练近 100×**

这种"过训练小模型"在 benchmark 上仍持续提升,且推理便宜。LLaMA 路线证明 **Chinchilla 是训练最优,不是 inference 最优**。

### 17.4.4 推论

| 用途 | 选型 |
| --- | --- |
| 一次性研究 / 闭源服务 | Chinchilla 比例 |
| 大规模长期推理 / 开源生态 | 过训练小模型 |
| 极端边缘部署 | 7B/3B + 极致过训练 |

---

## 17.5 Emergent Abilities(涌现)的争议

### 17.5.1 现象

Wei et al., 2022 报告:在某些 benchmark 上,模型尺寸 < $X$ 时表现近随机,> $X$ 时突然达到高分,称为 "**涌现能力**"。

### 17.5.2 反驳

Schaeffer et al., 2023:涌现往往是 **评测指标(exact match)的离散性** 造成的。换成连续指标(token-level log-prob)后,scaling 仍是平滑幂律。

### 17.5.3 共识

- 离散评测下涌现是真实观察
- 但底层 log-prob 改进是连续的
- 工程上选型可作"现象层"参考,但科学上不必神化"涌现"

---

## 17.6 MoE 与稀疏模型的 scaling

### 17.6.1 有效参数 vs 总参数

MoE 模型有两个参数概念(§08.8、§23):

- **总参数 $P_\text{total}$**:所有专家加起来
- **激活参数 $P_\text{act}$**:单 token 实际过的参数(Top-k × 单专家)

| 模型 | $P_\text{total}$ | $P_\text{act}$ |
| --- | --- | --- |
| LLaMA-3 70B(dense) | 70B | 70B |
| Mixtral 8×22B | 141B | 39B |
| DeepSeek-V3 | 671B | 37B |

### 17.6.2 MoE scaling 实验

Clark et al., 2022 / Krajewski et al., 2024 发现:**MoE 的 loss 介于 $P_\text{act}$ 与 $P_\text{total}$ 之间**,接近一个**有效参数** $P_\text{eff} \in (P_\text{act}, P_\text{total})$,通常 ~ $\sqrt{P_\text{total} \cdot P_\text{act}}$。

含义:**MoE 用比 dense 更少的激活算力,换接近 dense 的质量**——这是 671B / 37B 激活仍能与 405B dense 持平的根因。

### 17.6.3 训练算力

MoE 训练算力大约 $C \approx 6 P_\text{act} D$,显著小于按 $P_\text{total}$ 算。这就是"MoE 用 dense 1/10 训练成本"的来源。

### 17.6.4 Inference 算力

推理也按 $P_\text{act}$ 算 → MoE 是"训便宜 + 推便宜"的双优;代价是显存按 $P_\text{total}$(需装全部专家)。

---

## 17.7 数据约束 / 重复 token 的 scaling

### 17.7.1 高质量数据有上限

互联网英语高质 token 估计 ~ 30T(去重 + 过滤后)。LLaMA-3 已用 15T,接近天花板。

### 17.7.2 重复 token 的代价

Muennighoff et al., 2023:**重复 epoch 在前 4 epoch 内仍有效,> 4 epoch 边际收益骤降**。但比"完全不训"好得多。

### 17.7.3 Scaling 的瓶颈转移

未来的瓶颈可能不再是"算力",而是:

- **数据**:高质 token 用完
- **合成数据**:质量保障与塌缩(§11.11)
- **算法**:能不能"用更少数据训出同等模型"

---

## 17.8 Test-time Scaling(R1 / o1 路线)

### 17.8.1 现象

OpenAI o1 与 DeepSeek-R1 显示:**让模型在推理时输出更长的思维链(CoT),benchmark 仍可显著提升**。

### 17.8.2 新维度:推理 compute

把推理 token 数也作为一个 scaling 维度:

$$
\text{Accuracy} \approx f(P_\text{model}, P_\text{train}, P_\text{infer per query})
$$

DeepSeek-R1 论文显示:在数学题上,**推理 budget 翻倍,准确率随之上升**,符合幂律。

### 17.8.3 含义

- "更大模型"不是唯一路线;"更长推理"也能 scale
- 训练算力 vs 推理算力可"互换"——预算可以在两者间灵活分配
- §28 详细讨论 R1 路线

---

## 17.9 工程实用:给你一笔预算,怎么花?

### 17.9.1 步骤(以 $5M USD 为例)

1. **算云算力**:$5M ≈ 1.5M H100-小时 ≈ $5 \times 10^{23}$ FLOPs
2. **Chinchilla**:$C = 6PD$,$D = 20P$ → $P^* \approx \sqrt{C / 120} \approx 65B$, $D^* \approx 1.3 T$
3. **inference-aware**:若长期需要服务,把 $P$ 砍到 ~ 13B,$D$ 推到 4-15T
4. **MoE 路线**:若有专业团队,~ 200B 总参 / 30B 激活,可显著节省训练成本
5. **保险预留**:实际只用 70% 预算,剩 30% 留给 SFT/RLHF/迭代

### 17.9.2 各家公开配比

| 模型 | $P$ | $D$ | $D/P$ | 路线 |
| --- | --- | --- | --- | --- |
| GPT-3 | 175B | 300B | 1.7 | Kaplan |
| Chinchilla 70B | 70B | 1.4T | 20 | Chinchilla |
| LLaMA-2 7B | 7B | 2T | 286 | Inference-aware |
| LLaMA-3 8B | 8B | 15T | 1875 | 极端 inference-aware |
| Mistral-7B | 7B | ~8T | 1142 | Inference-aware |
| DeepSeek-V3 | 671B / 37B 激活 | 14.8T | (按 act 400) | MoE + Inference-aware |

### 17.9.3 给小团队的建议

- 不要从头训 → 微调一个开源模型
- 如果一定要训 → 用 $D/P \geq 100$,小而精
- 算力受限优先 LoRA 微调,而非预训练

---

## 17.10 完整代码:扫小模型估算 scaling

```python
"""
扫几个尺寸小模型 → 拟合 scaling law → 外推大模型预期 loss
"""

import numpy as np
from scipy.optimize import curve_fit

# (params, tokens, final_loss) — 真实数据填入
data = np.array([
    [  50e6,  5e9, 3.21],
    [  50e6, 10e9, 3.05],
    [ 125e6,  5e9, 2.92],
    [ 125e6, 10e9, 2.75],
    [ 350e6,  5e9, 2.66],
    [ 350e6, 10e9, 2.49],
])

P, D, L = data[:,0], data[:,1], data[:,2]

# Chinchilla 形式
def chinchilla(X, L_inf, A, alpha, B, beta):
    P, D = X
    return L_inf + A / P**alpha + B / D**beta

popt, _ = curve_fit(chinchilla, (P, D), L,
                    p0=[1.0, 100, 0.3, 100, 0.3], maxfev=20000)
L_inf, A, alpha, B, beta = popt
print(f"L_inf={L_inf:.3f} α={alpha:.3f} β={beta:.3f}")

# 外推:7B 模型 1T token 的预期 loss
P_new, D_new = 7e9, 1e12
L_pred = chinchilla((P_new, D_new), *popt)
print(f"7B / 1T token 预期 loss = {L_pred:.3f}")

# 给定 compute C,扫 P 找最优
C = 6 * 7e9 * 1e12   # 7B Chinchilla
Ps = np.logspace(8, 11, 50)
Ds = C / (6 * Ps)
Ls = chinchilla((Ps, Ds), *popt)
P_opt = Ps[np.argmin(Ls)]
print(f"在 C={C:.2e} 下最优 P ≈ {P_opt:.2e}")
```

这是真实研究的 baseline 流程:先小规模扫描,再外推大规模决策。

---

## 🎨 17.10.A 通俗类比合集

### 类比一: 烤面包配方 🍞

> 模型 = 面团大小,数据 = 烤箱时间,loss = 焦糊程度的反向。

- **Kaplan(2020)**:大面团,短时间 → 表面焦但里面生
- **Chinchilla(2022)**:中等面团,**长时间** → 内外金黄,$D \approx 20 P$ 是黄金比
- **LLaMA / inference-aware**:小面团,**超长时间** → 外卖外送方便,因为顾客每天都要吃
- **MoE**:**多面团并联**,顾客每次只吃一种,总体备料多但单次烤少
- **Test-time scaling**:面包做好后,客户**多嚼几下**(reasoning)也能尝出更多层次

### 类比二: 拍电影的预算分配 🎬

> 预算固定,导演决定怎么花。

- **演员阵容($P$)**:多大的明星
- **拍摄时长($D$)**:拍多少天
- **特效后期($P_\text{act}$ vs $P_\text{total}$)**:动用多少特效团队
- **Kaplan**:全砸明星(大模型,小拍摄)
- **Chinchilla**:明星与拍摄时间平衡
- **LLaMA 路线**:中等明星 + **超长拍摄**,因为这片会重复放映很久
- **Test-time**:首映夜让观众讨论 30 分钟(reasoning),口碑发酵

### 类比三: 餐厅扩张策略 🍜

> 你有 $5M,开几家什么样的餐厅?

- **一家旗舰大店**:Kaplan 风(大模型小数据)
- **多家中等门店,每家把菜单做到极致**:Chinchilla
- **大量小快餐店,每家供应同一份精心打磨菜单**:LLaMA-3(过训练小模型 + 海量推理)
- **MoE 餐厅**:**多个厨师专精不同菜系**,客人只点 2-3 道
- **Test-time**:让客人**点餐 + 自助沙拉吧 + 慢炖区**,把推理时间也当卖点

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🍞 烤面包 | **面团 vs 烤时** | $P$ vs $D$ |
| 🎬 电影 | **预算分配** | inference-aware 思路 |
| 🍜 餐厅 | **规模与重复** | MoE / test-time |

**记忆口诀**:
> **"Chinchilla 每参 20 token;过训练小模型推理省;MoE 训便宜推也便宜;test-time 是新维度。"**

---

## 17.11 常见疑问

### Q1: 我训练 7B 模型该用多少 token?
**A**: Chinchilla 推荐 140B;但实际开源圈普遍用 2T-15T(过训练 10-100×)。**用越多越好,只要不重复超 4 epoch**。算力允许下首选过训练。

### Q2: 我能在 1 张 4090 上自己扫 scaling law 吗?
**A**: 不能扫到大模型,但可以扫 100M-1B 验证形式。Chinchilla 论文的最小模型也才 70M——这正是 §17.10 的实践意义。

### Q3: 涌现到底真假?
**A**: 取决于 metric。离散 metric(exact match)上有突变;continuous metric(log-prob)上是平滑提升。工程层选型可用涌现作"现象层"指标,但别误以为是"魔法"。

### Q4: 数据用完了怎么办?
**A**: 当前路径:
- 合成数据(Phi 系列)
- 多 epoch(< 4)
- 高质量数据精选 + 退火(LLaMA-3)
- 多模态(用图像 / 视频补)
- test-time scaling(用推理 budget 替代训练 budget)

### Q5: 为什么 GPT-4 / Claude 不公布 P 和 D?
**A**: 商业秘密 + 法律风险(版权数据规模可能被审视)。但社区基于 cost 与 latency 可粗略推测(GPT-4 据信是 MoE,总 ~1.8T,激活 ~280B)。

---

## 17.12 本章小结

### 核心公式

**Chinchilla**:

$$
L(N, D) \approx L_\infty + A N^{-\alpha} + B D^{-\beta}, \quad D^* \approx 20 N^*
$$

**训练算力**:

$$
C \approx 6 P D
$$

**Inference-aware 总成本**:

$$
C_\text{total} = 6 P D + 2 P D_\text{infer}
$$

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **Scaling law** | loss = 幂律 + 不可约项 |
| **Chinchilla** | $D \approx 20 P$,训练最优 |
| **Inference-aware** | 过训练小模型,推理友好 |
| **MoE** | $P_\text{eff} \in (P_\text{act}, P_\text{total})$ |
| **Test-time scaling** | 推理 budget 是新维度 |

### 一句话总结

> **Scaling law 把 LLM 从"试错工程"变成"可预测工程":Chinchilla 给出训练最优 $D \approx 20P$,LLaMA 系列把"过训练小模型"做成 inference 最优,MoE 把激活参数与总参数解耦,o1/R1 又开辟 test-time scaling 维度——每一代 SOTA 都是在这些公式的某个新维度上做出最优配比。**

---

## 17.13 延伸阅读

### 必读论文

1. Kaplan et al., 2020. *"Scaling Laws for Neural Language Models"*. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
2. Hoffmann et al., 2022. *"Training Compute-Optimal Large Language Models"*(Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
3. Touvron et al., 2023. *"LLaMA"* / 2024 *"LLaMA-3"*(实证 inference-aware)

### 进阶论文

4. Wei et al., 2022. *"Emergent Abilities of Large Language Models"*. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)
5. Schaeffer et al., 2023. *"Are Emergent Abilities of Large Language Models a Mirage?"*. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
6. Muennighoff et al., 2023. *"Scaling Data-Constrained Language Models"*. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
7. Krajewski et al., 2024. *"Scaling Laws for Fine-Grained Mixture of Experts"*. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871)
8. DeepSeek-AI, 2025. *"DeepSeek-R1"*. (test-time scaling 实证)

### 推荐资源

- EpochAI: *"Compute trends in machine learning"*
- *"The Bitter Lesson"* — Rich Sutton

---

## 17.14 实战练习

### 练习 1 (★)

读 Chinchilla 论文 Table 3,验证公式 17.4 与原始数据的拟合。

### 练习 2 (★★)

在小数据上扫 (P, D):训 5 个 (P, D) 组合(P ∈ {50M, 125M, 350M}, D ∈ {5B, 10B}),用 §17.10 拟合幂律,与 LLaMA-2 7B 公开的 final loss 比较。

### 练习 3 (★★)

写脚本:给定预算 $C$(FLOPs),扫 $P$ 找最优 (P, D),分别在 "纯训练成本"与 "训练 + 1T inference" 两种目标下,比较最优 $P$ 的差异。

### 练习 4 (★★★)

研究 MoE scaling:把固定 $P_\text{act} = 7B$,扫 $P_\text{total} \in \{7B, 14B, 56B, 224B\}$,绘制 loss 曲线,验证"有效参数 $\sqrt{P_\text{total} \cdot P_\text{act}}$"的近似。

### 练习 5 (★★★)

复现 test-time scaling:对一个数学小数据集,改变最大 reasoning token 数 (256, 1024, 4096),绘制准确率 vs reasoning budget 的曲线。

---

## 章节交叉引用

- 前置: [§10 Block 整体](../Part_II_Transformer架构/10_Transformer_Block整体.md), [§11 数据准备](11_数据准备与清洗.md), [§13 训练循环](13_训练循环与混合精度.md)
- 后续: [§22 本地部署](../Part_IV_推理与部署/22_本地部署实战.md)(选型经济学), [§27 DeepSeek-V3 系统设计](../Part_V_DeepSeek专题/27_DeepSeek_V3_系统设计.md)
- 相关: [§23 DeepSeekMoE](../Part_V_DeepSeek专题/23_DeepSeekMoE.md), [§28 DeepSeek-R1](../Part_V_DeepSeek专题/28_DeepSeek_R1推理模型.md)

---

*Part III 完结。下一部分:Part IV 推理与部署。*
