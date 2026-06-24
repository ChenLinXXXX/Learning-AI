# 23 DeepSeek 系列演进(从 V1 到 V3.x / R1 的技术族谱)

> **章节定位**: Part V 起点,梳理 DeepSeek 技术族谱,后续 §24-28 各打一拳
> **预计阅读时间**: 50-70 分钟
> **难度**: ★★★☆☆
> **前置知识**: §07 Attention、§08 FFN/MoE、§10 Block 整体、§13 训练循环、§14 分布式
> **后续依赖**: §24 MLA、§25 DeepSeekMoE、§26 V3 工程、§27 V3 vs LLaMA-3、§28 R1

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [23.1 为什么单开一个"DeepSeek 专题"?](#231-为什么单开一个deepseek-专题)
- [23.2 公司与开源风格](#232-公司与开源风格)
- [23.3 时间线总览(2023–2026)](#233-时间线总览20232026)
- [23.4 V1 / Coder / Math:奠基阶段](#234-v1--coder--math奠基阶段)
- [23.5 V2:MLA + DeepSeekMoE 首登场](#235-v2mla--deepseekmoe-首登场)
- [23.6 V3:MTP + FP8 + DualPipe + 671B/37B](#236-v3mtp--fp8--dualpipe--671b37b)
- [23.7 R1 / R1-Zero:GRPO + 自发 reasoning](#237-r1--r1-zerogrpo--自发-reasoning)
- [23.8 V3.x 与后续](#238-v3x-与后续)
- [23.9 五大创新一句话总结](#239-五大创新一句话总结)
- [23.10 影响力:开源 SOTA 与"效率派"](#2310-影响力开源-sota-与效率派)
- [🎨 23.10.A 通俗类比合集](#-2310a-通俗类比合集)
- [23.11 常见疑问](#2311-常见疑问)
- [23.12 本章小结](#2312-本章小结)
- [23.13 延伸阅读](#2313-延伸阅读)
- [23.14 实战练习](#2314-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

DeepSeek 在 2024-2025 把开源 LLM 推向"质量与商业模型持平、训练成本却低一个量级"的新水位:**V3 用 ~$5.5M 训出对标 GPT-4 的开源模型,R1 用 GRPO 让 base 模型自发学会长思维链**。本章先给出"族谱图",后续 §24-28 分别深入每一项核心创新:

- **§24 MLA**:把 KV cache 压 100×+,长上下文经济性的根
- **§25 DeepSeekMoE**:细粒度专家 + 共享专家 + 负载均衡 loss
- **§26 V3 训练工程**:FP8 + DualPipe + 通信计算重叠
- **§27 V3 vs LLaMA-3**:同代旗舰对照
- **§28 R1**:GRPO + 可验证奖励 + reasoning 涌现

读完本章你拿到一张"DeepSeek 心智地图",再去读后面任一章不会迷路。

---

## 学习目标

完成本章后,你能够:

- ✅ 列出 V1 → V3 → R1 的核心技术演进 (§23.3-7)
- ✅ 解释"为什么 MLA + MoE 不是孤立技术,而是 V2/V3 系统设计的对偶面" (§23.5-6)
- ✅ 区分 V3 与 R1 的训练目标差异 (§23.7)
- ✅ 一句话说出 DeepSeek 的设计哲学(效率派 vs 蛮力派) (§23.10)

---

## 23.1 为什么单开一个"DeepSeek 专题"?

LLM 开源圈里,DeepSeek 的特别之处在于它**同时**推动了三件事:

1. **架构创新**:MLA 与 DeepSeekMoE 显著改变了 KV 与 FFN 两条主线
2. **训练系统**:FP8、DualPipe 把硬件 / 通信压到接近极限
3. **能力解锁**:R1 用纯 RL 让 base 模型涌现 reasoning

任一项都值得一个专题。它们又是同一支团队、同一套工程体系演进出来——单章讲不清,所以单开 Part。

> 这一部分不重复前 4 部分的"通用知识",只讲 DeepSeek 怎么把它们组合 / 改造。

---

## 23.2 公司与开源风格

DeepSeek(深度求索)由幻方量化(High-Flyer)2023 年创立。开源风格:

- **真正可商用**:MIT/类 MIT 协议,无 LLaMA 那种"7 亿月活上限"条款
- **技术报告极详细**:V2 / V3 / R1 都是 30+ 页论文体,包含训练曲线、loss、ablation
- **基础设施开源**:推理 kernel(DeepGEMM)、训练框架(DeepEP)、通讯库(DualPipe 后续)逐步发布
- **量化与服务**:模型卡 + 官方 API + 第三方主流引擎(vLLM / SGLang)很快支持

对中文社区来说,DeepSeek 与 Qwen(阿里) 是国内 LLM 的两大开源旗手。

---

## 23.3 时间线总览(2023–2026)

| 时间 | 发布 | 关键贡献 |
| --- | --- | --- |
| 2023.11 | **DeepSeek LLM 7B / 67B** | 第一代 dense 模型,数据 2T,LLaMA-2 同代 |
| 2024.01 | **DeepSeek-Coder 系列** | 代码模型,16K 上下文,FIM 训练 |
| 2024.02 | **DeepSeekMath 7B** | 数学专项,首次公开提出 **GRPO** |
| 2024.05 | **DeepSeek-V2 236B / 21B 激活** | 首次 **MLA + DeepSeekMoE**,128k 上下文 |
| 2024.07 | **DeepSeek-V2.5** | 整合 V2 + Chat + Coder,服务统一 |
| 2024.12 | **DeepSeek-V3 671B / 37B 激活** | **FP8 训练 + MTP + DualPipe**,对标 GPT-4 |
| 2025.01 | **DeepSeek-R1 / R1-Zero** | **GRPO + 可验证奖励**,reasoning 涌现 |
| 2025.中 | **R1 蒸馏小模型(1.5B-70B)** | 把 R1 reasoning 灌到 Qwen / LLaMA 小尺寸 |
| 2025+ | V3.1 / V3.2 / R 系列迭代 | 长上下文、tool use、对齐持续优化 |

技术演进可以归纳为三波:
- 第一波(V1 / Coder / Math):跟随 + 在数学/代码上突破
- 第二波(V2 / V3):**架构创新主导**(MLA + MoE)
- 第三波(R1 / R1-Zero):**训练范式创新**(RL 单独驱动 reasoning)

---

## 23.4 V1 / Coder / Math:奠基阶段

### 23.4.1 DeepSeek LLM 7B / 67B(2023.11)

- 标准 LLaMA 风格 dense:RMSNorm + SwiGLU + RoPE + MHA/GQA
- 数据 2T token,中英混合,代码 + 数学比例较高
- 67B 在评测上紧追 LLaMA-2 70B,中文显著更强

**意义**:证明国产开源团队能做出"齐平 LLaMA-2"的模型,为后续创新打好基础设施。

### 23.4.2 DeepSeek-Coder(2024.01)

- 16K 上下文(当时较激进)
- FIM(§12.4)训练 50% 比例
- 在 HumanEval / MBPP / DS-1000 上接近 GPT-3.5

特别有意义的是**代码数据流水线**(用 GitHub + 严格 license 过滤 + 代码语义去重),后续 V3 复用其中很多 pipeline。

### 23.4.3 DeepSeekMath(2024.02)

- 数学专项 7B,在 MATH benchmark 上首次开源 30+%
- 论文首次提出 **GRPO**(§16.7):group-relative policy optimization,作为 PPO 的轻量替代
- 用可验证奖励(答案匹配 / 单位检查)做 RL

**回望**:GRPO 在这里只是个数学专项工具,**一年后却成了 R1 的核心**。这是技术演化里典型的"暗线"。

---

## 23.5 V2:MLA + DeepSeekMoE 首登场

### 23.5.1 一句话定位

> V2 = "在同等推理算力下,用 21B 激活打到 dense 70B 的质量,且 128k 上下文几乎不要 KV cache"

### 23.5.2 关键技术

1. **MLA(Multi-head Latent Attention,§24)**:把 KV cache 压到 5-10% of MHA
2. **DeepSeekMoE(§25)**:细粒度专家(160 个)+ 共享专家,每 token Top-6 激活
3. **128k 上下文**:训练 32k + YaRN 扩到 128k
4. **辅助 loss 负载均衡**:防止专家偏斜

### 23.5.3 数字

- 总参 236B,激活 21B
- 训练 token 8.1T
- 训练成本约 \\$2.5M(对照 LLaMA-3 70B ~ \\$50M+)
- Benchmark:接近 LLaMA-3 70B,部分中文超越

### 23.5.4 为什么 MLA + MoE 是"对偶"?

- MLA 压住 KV cache → 长上下文 / 高并发的显存瓶颈被解
- MoE 压住激活算力 → 大参数模型推理 / 训练 cost 都受控
- 两者刚好打在 LLM 的两大成本头(attention 显存 + FFN 算力),协同收益最大

这套组合在 V3 中被推到极致。

---

## 23.6 V3:MTP + FP8 + DualPipe + 671B/37B

### 23.6.1 一句话定位

> V3 = "把 V2 的架构再 ×3,加 MTP / FP8 / DualPipe 工程,达到 GPT-4 级别质量,只用 ~\$5.5M 训练费"

### 23.6.2 关键技术

| 技术 | 详见章节 |
| --- | --- |
| MLA(继承) | §24 |
| DeepSeekMoE(扩展:256+1 专家,Top-8) | §25 |
| **MTP** 辅助 loss | §12.5 / §26 |
| **FP8 训练** | §13.3.5 / §26 |
| **DualPipe** 双向流水线 | §14.6.4 / §26 |
| 拓扑感知 All-to-All | §14.8 / §26 |

### 23.6.3 数字

- 总参 671B,激活 37B(MoE,256 专家 + 1 共享)
- 训练 14.8T token,bf16 + FP8 混合精度
- 训练成本约 \\$5.5M(2048 张 H800,~2 个月)
- 上下文 4k 预训练 + YaRN 扩到 128k
- Benchmark:在 MMLU/GPQA/MATH/HumanEval 上与 GPT-4o / Claude 3.5 Sonnet 大体并驾齐驱

### 23.6.4 与 Chinchilla / Inference-aware 的关系

D/P 按激活 ≈ 400,远超 Chinchilla 推荐 ~20——是**典型的 inference-aware 路线**(§17.4)。但因为 MoE,**总参数其实是 dense 的 ~10×**,有效参数 $\sqrt{P_\text{total} P_\text{act}} \approx 150B$,质量与 dense 100B+ 持平。

### 23.6.5 为什么这件事重要?

公布前业界普遍认为"GPT-4 级开源模型需要 \\$50-100M 训练"。V3 把这个数字降一个数量级,**直接重设了开源 LLM 的成本预期**。

---

## 23.7 R1 / R1-Zero:GRPO + 自发 reasoning

### 23.7.1 一句话定位

> R1 = "在 V3-Base 上用 GRPO 训练 reasoning,模型自发学会长思维链,效果对标 o1"

### 23.7.2 两种范式

| 版本 | 训练范式 |
| --- | --- |
| **R1-Zero** | 直接在 V3-Base 上跑 GRPO,**完全跳过 SFT**,reward = 数学答案正确性 + 格式规范 |
| **R1** | SFT(高质量 reasoning data) → GRPO → 蒸馏 → 二次 GRPO 的多阶段 |

R1-Zero 的发现尤为震撼:**纯 RL 即可让模型自发涌现思维链**(`<think>...</think>`)。这是 LLM 历史上少见的"涌现"实证。

### 23.7.3 蒸馏成果

DeepSeek 把 R1 的 reasoning 数据 + 输出蒸馏到 Qwen / LLaMA 1.5B–70B,**这些小模型在数学 / 代码上反超原模型 + GPT-4o**——证明 reasoning 可"通过数据"被压缩进小模型。

### 23.7.4 对行业的影响

- 开源圈第一次跑通"o1 路线"
- GRPO 作为简化 RL 算法被广泛复用(Qwen / OpenAI 后续路线均有 GRPO 影子)
- 把"test-time scaling"(§17.8)从猜想变实证

R1 是 2025 年开源 LLM 最重要的事件,与 V3 在不同维度上各打一拳。

---

## 23.8 V3.x 与后续

2025 中-2026 期间 DeepSeek 持续迭代:

- V3.1 / V3.2 / V3.3:长上下文 + tool use + 对齐改进
- R 系列升级:更多领域 reasoning 支持(代码、科学、医学)
- 内部推理栈开源化(DeepEP、DeepGEMM 等 kernel)

具体数字与论文随时间变化,本书仅记录到 V3 + R1 两座里程碑作为锚点。

---

## 23.9 五大创新一句话总结

| 技术 | 一句话 | 详见 |
| --- | --- | --- |
| **MLA** | 把 KV cache 压成低秩潜变量,128k 上下文仍小 | §24 |
| **DeepSeekMoE** | 256 个细粒度专家 + 1 共享专家 + Top-8 路由 + 负载 loss | §25 |
| **MTP** | 训练时多 head 预测多 token,推理时作 speculative draft | §12.5 / §26 |
| **FP8 训练** | per-tile 缩放的 E4M3/E5M2 混合,2× 吞吐 / 0.6× 显存 | §13.3.5 / §26 |
| **GRPO** | 用组平均 reward 做 baseline 的轻量 PPO,适合可验证奖励 | §16.7 / §28 |

---

## 23.10 影响力:开源 SOTA 与"效率派"

把全球开源 LLM 团队大致分两派:

- **蛮力派(Meta LLaMA / Mistral 大模型)**:大数据 + 大模型 + 大算力,工程稳重
- **效率派(DeepSeek / Qwen / 部分国内团队)**:架构创新 + 系统极致优化,**每美元换更多质量**

DeepSeek 是效率派代表,影响在多个层面:

1. **架构**:MLA 在 2024-2025 被大量复现/类似设计采用(Kimi K1.5、Moonshot 等)
2. **系统**:DualPipe / FP8 等成为大模型团队的标配研究方向
3. **训练范式**:GRPO 几乎成为 reasoning RL 的事实默认
4. **商业**:把"开源 vs 闭源"的成本差异从 10-100× 砍到 ~1-2×,改变了部分公司的 build vs buy 决策

---

## 🎨 23.10.A 通俗类比合集

### 类比一: 汽车工业的"丰田时刻" 🚗

> DeepSeek 之于 LLM,有些像丰田之于汽车。

- 美式肌肉车(GPT-4 / LLaMA 70B):大引擎、大马力、大油耗
- 日式精益车(DeepSeek-V3):同等动力,**油耗一半**,因为发动机 + 变速箱 + 空气动力都重新设计
- **MLA** = 重新设计变速箱(更高效传动)
- **MoE** = 智能气缸(只激活需要的几缸,省油)
- **FP8** = 新材料制造,轻量化
- **GRPO** = 自学习驾驶员,自己摸索最佳赛道路线

### 类比二: 厨房的"低成本米其林" 🍽️

> 同样一份米其林三星菜单,有两种做法。

- 蛮力派:头牌大厨 + 顶级食材 + 4 小时慢炖
- DeepSeek:**改流程**——分工细化、关键步骤一次到位、不浪费一份食材
- **MLA** = 高效切配工序,备料量少但菜出得快
- **MoE** = 不雇 50 个全才大厨,而是 256 个专精厨师,每桌只叫合适的 8 个
- **MTP** = 主厨边炒边把"下一步" / "再下一步"也提前预备
- **GRPO 自学** = 没有顶级老师,小厨子自己反复尝试 + 食客打分,**自发学会创意菜**

### 类比三: 编程语言的"开源精神" 🐧

> DeepSeek 像 Linux 之于操作系统。

- 闭源派(OpenAI / Anthropic):内部精密黑盒
- DeepSeek:**完整公开训练流程 + 论文 + 模型权重 + 部分 kernel 代码**
- 让全球研究者能"看清细节、动手复现",开源生态获得跃迁动力

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🚗 丰田 | **效率重设计** | MLA / MoE / FP8 协同 |
| 🍽️ 米其林 | **流程优化** | 训练成本砍到 1/10 |
| 🐧 Linux | **开源生态** | 技术报告 + 代码全开 |

**记忆口诀**:
> **"MLA 压 KV,MoE 压算力,FP8 压精度,DualPipe 压通信,GRPO 解锁 reasoning。"**

---

## 23.11 常见疑问

### Q1: DeepSeek 真比 OpenAI 便宜 10×?
**A**: 训练成本对比公开数字属实(V3 ≈ \\$5.5M, GPT-4 估算 \\$100M+)。但比较口径不完全公平:
- 不含 SFT/RLHF/数据采集成本
- 不含失败实验成本
- GPU 价格各异
仍可说**至少便宜数倍**,且**架构与系统优化是真实可复现的**。

### Q2: V3 与 GPT-4o 谁强?
**A**: 看维度。
- 中文 / 数学 / 代码:V3 略强或持平
- 多模态 / tool use / "魅力体验":GPT-4o 通常更稳
- 推理 (CoT):V3 + R1 路线非常强

### Q3: 我要不要等 V4?
**A**: 已经稳定的 V3 / R1 + 第三方推理框架已经足够生产。等新版的风险是 release 节奏不可预测。

### Q4: MLA 与 MoE 必须一起用吗?
**A**: 不必。理论上两者正交,可独立采用。Mistral 系列 GQA + MoE(无 MLA);Qwen 部分版本 MLA-like + dense。DeepSeek 是把两者协同推到极致的代表。

### Q5: R1 训练能在小团队复现吗?
**A**: 部分可。R1 蒸馏的小模型(7B / 14B)可在几张 A100 / H100 上 SFT + GRPO 复现 reasoning 风格;但 V3-Base 量级的 GRPO 仍需大集群。

---

## 23.12 本章小结

### 演进主线

```text
V1 / Coder / Math   →  V2(MLA + MoE)        →  V3(+MTP / FP8 / DualPipe)
                                                  │
                                                  ▼
                                              R1-Zero(纯 GRPO)
                                                  │
                                                  ▼
                                              R1(SFT + GRPO 多阶段 + 蒸馏)
```

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **效率派代表** | 同质量,1/10 训练成本 |
| **架构创新** | MLA + DeepSeekMoE |
| **系统创新** | FP8 + DualPipe + EP |
| **范式创新** | GRPO + 可验证奖励 + reasoning 涌现 |
| **开源贡献** | 详细技术报告 + 工具链 |

### 一句话总结

> **DeepSeek 是 2024-2025 开源 LLM 的效率派旗手:架构上用 MLA 与 DeepSeekMoE 把"长上下文 + 大参数"的成本拍扁,系统上用 FP8 与 DualPipe 把训练费砍到 GPT-4 的 1/10,范式上用 GRPO 让纯 RL 自发涌现 reasoning——后续 §24-28 分别打开这五项创新的工程细节。**

---

## 23.13 延伸阅读

### 必读论文

1. DeepSeek-AI, 2024. *"DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"*. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)
2. DeepSeek-AI, 2024. *"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"*. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
3. DeepSeek-AI, 2024. *"DeepSeek-V3 Technical Report"*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
4. DeepSeek-AI, 2025. *"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
5. Shao et al., 2024. *"DeepSeekMath"*(GRPO 提出场)[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

### 进阶资源

6. DeepSeek-Coder 论文
7. *"DeepSeek-V3 vs GPT-4 vs Llama-3"* 各类社区评测博客
8. SemiAnalysis / Stratechery 等关于 DeepSeek 商业模型的分析

---

## 23.14 实战练习

### 练习 1 (★)

读 V3 technical report Abstract + Section 1,总结**最重要的 5 个数字**(参数 / 激活 / token / 成本 / 训练时长)。

### 练习 2 (★)

打开 R1 论文,定位 R1-Zero 训练过程中"thinking 长度变化"的图,描述其趋势。

### 练习 3 (★★)

在本地用 Ollama 跑 `deepseek-r1:7b`(蒸馏版),用同一题(GSM8K)对比 base 7B 模型与 R1-蒸馏 7B 的输出长度与正确率。

### 练习 4 (★★)

绘制一张 DeepSeek 技术家族图(包含 V1/Coder/Math/V2/V2.5/V3/R1-Zero/R1/蒸馏),标注每个节点的"核心创新"。

### 练习 5 (★★★)

设计一个 ablation 实验设想:固定数据与算力,只切换"V2 架构 vs V3 架构",预测哪几个 benchmark 改进最显著,为什么。读 V3 论文 verify 你的猜测。

---

## 章节交叉引用

- 前置: [§08 FFN / MoE](../Part_II_Transformer架构/08_FFN与激活函数.md), [§13 训练循环](../Part_III_训练原理/13_训练循环与混合精度.md), [§14 分布式](../Part_III_训练原理/14_分布式训练.md), [§16 偏好对齐](../Part_III_训练原理/16_偏好对齐.md)
- 后续: [§24 MLA](24_MLA深度解析.md), [§25 DeepSeekMoE](25_DeepSeekMoE.md), [§26 V3 训练工程](26_V3_训练曲线与工程.md), [§27 V3 vs LLaMA-3](27_V3_vs_LLaMA3.md), [§28 R1 推理路线](28_R1_推理路线.md)
- 相关: [§17 Scaling Law](../Part_III_训练原理/17_Scaling_Law.md), [§22 本地部署](../Part_IV_推理与部署/22_本地部署实战.md)

---

*下一章: §24 MLA 深度解析*
