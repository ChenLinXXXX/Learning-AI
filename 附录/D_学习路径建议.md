# 附录 D — 学习路径建议

> **范围**: 细化 README 已有的 4 条学习路径,**给出每条路径的"目标读者 → 每周节奏 → 章节顺序 → 输出物 → 评估方法"**。
> **用法**: 个人或读书会按表 plan;按完成度自测进度。

---

## 目录

- [D.0 共同前提](#d0-共同前提)
- [D.1 路径 1:系统入门(零基础到工程级,6 周)](#d1-路径-1系统入门零基础到工程级6-周)
- [D.2 路径 2:工程师快速通道(已有 ML 基础,2-3 周)](#d2-路径-2工程师快速通道已有-ml-基础2-3-周)
- [D.3 路径 3:面试与原理深化(2 周)](#d3-路径-3面试与原理深化2-周)
- [D.4 路径 4:部署优先(实战导向,1-2 周)](#d4-路径-4部署优先实战导向1-2-周)
- [D.5 路径 5:DeepSeek 专精(1 周,既有 LLM 基础者)](#d5-路径-5deepseek-专精1-周既有-llm-基础者)
- [D.6 路径 6:Agent / RAG 应用开发(1-2 周)](#d6-路径-6agent--rag-应用开发1-2-周)
- [D.7 学习方法 Tips](#d7-学习方法-tips)

---

## D.0 共同前提

- **环境**:一台能跑 Python 3.10+ 的机器;尽量有 GPU(消费级 12GB+ 即可大部分练习,部分要求云 A100/H100)
- **必装**:`pytorch`、`transformers`、`datasets`、`accelerate`、`peft`、`vllm` 或 `ollama`、`numpy`、`matplotlib`
- **配套**:打印一份 [附录 A 名词字典](A_名词字典.md) 与 [附录 B 公式速查](B_公式速查.md),贴在手边
- **节奏**:每章 60-90 分钟读 + 1-2 小时练习,周末 1 个综合项目

---

## D.1 路径 1:系统入门(零基础到工程级,6 周)

**适合**:扎实学过高数 / 线代,但没系统接触 NLP / Transformer 的工程师 / 学生。
**目标**:6 周后能独立读 V3 论文、跑通 GPT-mini 训练 + 部署。

| 周 | 章节 | 重点 | 输出物 |
| --- | --- | --- | --- |
| **W1** | §00 + §01 + §02 | 数学语言搭骨架 | 完成 §01 / §02 全部 ★ 练习 |
| **W2** | §03 + §04 + §05 | 反向传播 + 优化 + tokenizer | 手写 micrograd,跑通 XOR;BPE mini |
| **W3** | §06 + §07 + §08 | 嵌入 + 注意力 + FFN | 从零搭一层 Transformer Block |
| **W4** | §09 + §10 | 归一化 + Block 整体 | 跑通 GPTMini on tiny shakespeare |
| **W5** | §11 + §12 + §13 + §14(浏览) | 数据 + 训练 + 分布式概览 | 实际跑一次 SFT(LoRA on 7B) |
| **W6** | §17 + §18 + §22 + 选读 §15/§16 | scaling + 推理 + 部署 | 用 vLLM 起 endpoint + 配合 Open WebUI |

**评估**:能独立向同事/朋友讲清 Transformer 原理 + 跑通一次"训练 → 部署"链路。

---

## D.2 路径 2:工程师快速通道(已有 ML 基础,2-3 周)

**适合**:做过 CV / NLP 项目,会用 PyTorch,想快速补 LLM 知识。
**目标**:2-3 周补齐当代 LLM 全栈,能对 V3 / R1 论文逐节理解。

| 周 | 章节 | 备注 |
| --- | --- | --- |
| **W1** | §05-§10 + 跳过 Part I 大部分 | 集中突破 Transformer 架构 |
| **W2** | §11-§17 | 训练原理一遍过 |
| **W3** | §18-§22 + §23-§28(精读) | 推理 + DeepSeek 专题 |

**作业**:
1. 复现 nanoGPT 训练(W1)
2. 用 LoRA SFT 一个 7B 模型(W2)
3. 部署 + 自评 benchmark(W3)
4. 读完 R1 论文 + 写 500 字摘要(W3 末)

---

## D.3 路径 3:面试与原理深化(2 周)

**适合**:LLM 面试准备(算法岗 / SDE 转 ML)。
**目标**:能在白板上推导主要公式,讨论架构权衡。

| 主题 | 章节 | 必背 / 必懂 |
| --- | --- | --- |
| 注意力 | §07 + §24 | scaled dot-product, $\sqrt{d_k}$ 来源,causal mask,MHA/GQA/MLA 区别 |
| 位置编码 | §06 | sinusoidal / RoPE / ALiBi 公式与外推 |
| FFN / 归一化 | §08 / §09 | SwiGLU 形式,$8/3 d$ 由来,LN vs RMSNorm |
| 反向传播 | §03 | softmax+CE 梯度 $p-y$ |
| 优化 | §04 | AdamW 公式,Warmup+Cosine,Adam 显存 ≈ 12P |
| 训练 / 分布式 | §13 / §14 | bf16/FP8 取舍,ZeRO 1/2/3,TP/PP/EP |
| Scaling Law | §17 | Chinchilla 20:1,inference-aware,MoE 有效参数 |
| 推理 | §18 / §19 / §21 | Prefill/Decode,KV cache 公式,Flash/Speculative |
| 偏好对齐 | §16 | RLHF 三阶段,DPO 公式,GRPO 与 R1 |
| RAG / Agent | §30 / §31 | ReAct,Function Calling,Hybrid+Rerank |

**练习**:每章用 30 分钟做一份"手写笔记 + 一题白板推导";找搭档互讲。

---

## D.4 路径 4:部署优先(实战导向,1-2 周)

**适合**:产品 / 后端工程师 / 创业开发者,目标:本地或私有云能稳定服务 LLM。
**目标**:1-2 周内从零到上线一个 OpenAI 兼容 endpoint + RAG 应用。

| 阶段 | 章节 | 输出 |
| --- | --- | --- |
| **Day 1-2** | §17(选型经济学)+ §22(部署) | 估算硬件成本,选型;装 Ollama 跑通 |
| **Day 3-4** | §18 + §19 + §20 | 理解 prefill/decode,KV cache 估算;调采样参数 |
| **Day 5-6** | §21(量化 + Flash + Spec) | vLLM 上 AWQ-INT4 部署 7B / 13B / 70B |
| **Day 7-9** | §30 RAG | 接入 bge-m3 + reranker + HyDE,搭一个企业 RAG |
| **Day 10-12** | §31 Agent + §32 Prompt | 写 ReAct / FC Agent,做 Prompt 测试集 |
| **Day 13-14** | 上线 + 监控 + 安全 | nginx 鉴权,Prometheus 监控,prompt injection 防护 |

**最终产出**:一个能稳跑的小服务(企业问答 / 写作助手 / 客服 bot),配 README + 压测报告。

---

## D.5 路径 5:DeepSeek 专精(1 周,既有 LLM 基础者)

**适合**:已熟悉 Transformer + 训练,想集中突破 V3 / R1 工程细节。
**目标**:精读 V3 / R1 论文,能详尽讲解每个工程创新。

| 天 | 章节 + 论文 |
| --- | --- |
| **D1** | §23 演进 + V1/Math/V2 论文摘要 |
| **D2** | §24 MLA + V2 论文 §2 / V3 §2 |
| **D3** | §25 DeepSeekMoE + DeepSeekMoE 论文 |
| **D4** | §26 训练工程 + V3 §3 FP8 / §3.2 DualPipe |
| **D5** | §28 R1 + R1 论文 + DeepSeekMath GRPO |
| **D6** | §27 vs LLaMA-3 + Meta 的 *Llama 3 Herd* |
| **D7** | 综合复盘 + 给同事/读书会做一份 30 min talk |

**练习**:
- 手推 MLA 矩阵吸收的等价性
- 在小 MoE 上实现"auxiliary-loss-free" 路由
- 用 R1 蒸馏数据 SFT 一个 Qwen-1.5B

---

## D.6 路径 6:Agent / RAG 应用开发(1-2 周)

**适合**:已有 chat 模型可用(API 或本地),目标:把 LLM 真正变成"会干活"的应用。
**目标**:1-2 周内做出一个完整 Agent 产品原型。

| 阶段 | 章节 | 任务 |
| --- | --- | --- |
| **Day 1** | §20 + §32 | 学采样 + Prompt 四件套,做 prompt 测试集 |
| **Day 2-3** | §30 RAG | 跑通 RAG demo + 加 Hybrid + Reranker |
| **Day 4** | §31.2-3 | 写 ReAct 与 Function Calling 两种 Agent |
| **Day 5** | §31.4-5 | 加 Planning + Memory(短/长/工具) |
| **Day 6** | §31.7-8 | 接 MCP 或 Browser Use |
| **Day 7-10** | 综合 | 选一个真实问题(简历分析 / 代码 review / 客服)做端到端 Agent |

**产出**:Github 上能跑的 Agent demo,带 README + 评测集。

---

## D.7 学习方法 Tips

### D.7.1 教学三层法

> 看 → 写 → 教。

- **看**:读章节
- **写**:做练习,**亲手敲代码**(不要复制)
- **教**:用 5 分钟把核心思想讲给同事或朋友;讲不清就回去再读

### D.7.2 一句话总结习惯

读完每章 / 每篇论文,**强制写 1-2 句"一句话总结"**(本书每章已示范)。这是高密度知识的最佳压缩。

### D.7.3 公式 → 代码 → 直觉

抽象公式 → 找/写最少代码验证 → 用一个生活类比记住。三者一起作用最稳。

### D.7.4 不要陷入完美

- 不要等"把数学全学完"再读 Transformer
- 不要等"环境配完美"再写代码
- 不要等"读完所有论文"再实战

**"60% 准备就开始"** 是最有效路径。

### D.7.5 持续更新

LLM 领域半年大变。建议:

- 订阅 arXiv-sanity 或 Hugging Face Papers
- 关注 LMSys / OpenCompass leaderboard
- 看 SemiAnalysis / The Information / 36氪 等趋势分析
- 每月精读 1-2 篇 SOTA 论文,与本书章节对照

### D.7.6 学习社区

- HuggingFace 论坛 / Discord
- Reddit r/LocalLLaMA
- DeepSeek / Qwen / vLLM 官方 Discord
- 国内: 飞桨 / ModelScope 社区

---

*附录全部完成。整个项目主线写作至此结束。*
