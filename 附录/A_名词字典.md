# 附录 A — 完整名词字典

> **范围**: 整合本书 §01–§32 出现过的关键术语,**中英对照 + 一句话解释 + 章节回链**。
> **用法**: 阅读中遇到陌生词来这里快速定位;或作为复习速查表。
> **排序**: 按主题分组,组内按英文字母。

---

## 目录

- [A.1 数学与基础](#a1-数学与基础)
- [A.2 Transformer 架构](#a2-transformer-架构)
- [A.3 训练原理](#a3-训练原理)
- [A.4 推理与部署](#a4-推理与部署)
- [A.5 DeepSeek 专题](#a5-deepseek-专题)
- [A.6 应用前沿](#a6-应用前沿)
- [A.7 数值精度与硬件](#a7-数值精度与硬件)

---

## A.1 数学与基础

| 英文 | 中文 | 一句话解释 | 章节 |
| --- | --- | --- | --- |
| Backpropagation | 反向传播 | 用链式法则在计算图上倒序求梯度 | §03 |
| Cosine Similarity | 余弦相似度 | 衡量向量方向相近度的内积归一化 | §01.4.2 |
| Cross-Entropy | 交叉熵 | 用 q 编码 p 样本的平均码长,LLM loss 主形式 | §02.5.2 |
| Eigenvalue / Eigenvector | 特征值 / 特征向量 | 矩阵作用下只伸缩不旋转的方向 | §01.6 |
| Entropy | 熵 | 一个分布的"平均不确定性"(以 bit / nat 度量) | §02.5.1 |
| Gradient | 梯度 | 标量关于向量参数的偏导组成的同形张量 | §03.2 |
| Jacobian | 雅可比矩阵 | 向量函数对向量的偏导矩阵 | §03.2.2 |
| KL Divergence | KL 散度 | 两分布间的非对称"距离",训练 / 蒸馏核心工具 | §02.5.3 |
| Likelihood / MLE | 似然 / 最大似然估计 | 给数据找最可能参数,等价 cross-entropy | §02.4 |
| Norm(L2) | 范数 | 向量长度,用于裁剪 / 距离 | §01.4 |
| Perplexity(PPL) | 困惑度 | $\exp(\text{CE})$,平均每步"在多少 token 中犹豫" | §02.5.5 |
| Probability Density | 概率密度 | 连续随机变量的"单位长度概率" | §02.1 |
| SVD | 奇异值分解 | 任意矩阵 = U Σ V^T,低秩近似的根 | §01.5 |
| Softmax | 归一化指数 | 把任意实数向量映成合法概率分布 | §02.6 |
| Tensor | 张量 | 多维数组,LLM 计算的基本数据结构 | §01.3 |

---

## A.2 Transformer 架构

| 英文 | 中文 | 一句话解释 | 章节 |
| --- | --- | --- | --- |
| Attention | 注意力 | 用 Q·K^T 做相似度,softmax 后加权 V | §07 |
| ALiBi | 线性位置偏置 | attention 分数上加 -m·|i-j|,天然外推 | §06.6 |
| BBPE | 字节级 BPE | 在 UTF-8 字节上跑 BPE,零 OOV | §05.4 |
| Causal Mask | 因果掩码 | 让位置 t 只看 ≤ t,实现自回归 | §07.5 |
| Decoder-Only | 仅解码器 | LLM 主流架构,N 层堆叠 | §10.1 |
| Embedding | 嵌入 | token ID → 稠密向量 | §06.1 |
| FFN | 前馈网络 | Block 内逐 token 深加工模块 | §08 |
| GELU | 高斯误差线性单元 | $x \cdot \Phi(x)$ 平滑激活 | §08.3.2 |
| GQA | 分组查询注意力 | Head 分组共享 K/V,KV cache 中等压缩 | §07.9 |
| LayerNorm / RMSNorm | 层归一化 / RMS 归一化 | 沿特征维归一,RMSNorm 去均值更省 | §09 |
| MHA / MQA / MLA | 多头 / 多查询 / 潜变量注意力 | KV 压缩从 head 共享到低秩压缩 | §07.9 / §24 |
| MoE | 专家混合 | FFN 拆成 N 个专家,每 token Top-k 激活 | §08.8 / §25 |
| Pre-Norm / Post-Norm | 前 / 后归一化 | Norm 在子层入口或出口,Pre 更稳 | §09.5 |
| Residual Connection | 残差连接 | x + f(x),给梯度高速公路 | §09.2 |
| RoPE | 旋转位置编码 | 把位置编码做成 Q/K 的旋转,内积只看相对距离 | §06.5 |
| Self-Attention | 自注意力 | Q/K/V 全来自同一输入序列 | §07 |
| SiLU / Swish | Sigmoid 线性单元 | $x \cdot \sigma(\beta x)$ 平滑激活 | §08.3.3 |
| SwiGLU | SwiGLU 门控 | $\text{Swish}(W_g x) \odot (W_u x)$,LLaMA 默认 FFN | §08.4 |
| Tokenization | 分词 | 文本 → 整数 token 序列 | §05 |
| YaRN | RoPE 频率分段缩放 | 长上下文外推方案 | §06.7 |

---

## A.3 训练原理

| 英文 | 中文 | 一句话解释 | 章节 |
| --- | --- | --- | --- |
| AdamW | 解耦权重衰减 Adam | LLM 训练事实默认优化器 | §04.5 |
| BPE / WordPiece / Unigram | BPE / WordPiece / Unigram | 三种 subword 算法 | §05 |
| Chinchilla | Chinchilla 法则 | D ≈ 20·N 训练最优 | §17.3 |
| CLM / NTP | 因果建模 / 下一 token 预测 | LLM 主预训练目标 | §12.2 |
| Compute-Optimal | 算力最优 | 给定 C,选 (P, D) 让 loss 最小 | §17.3-4 |
| DPO | 直接偏好优化 | 无需 RM/PPO,直接在偏好对上优化 | §16.6 |
| DualPipe | 双向流水线 | PP forward / backward 同时双向流动,DeepSeek-V3 | §14.6 / §26.6 |
| EMA | 指数滑动平均 | 维护权重平滑版,推理偶用 | §13.6.4 |
| Emergent Abilities | 涌现能力 | 规模到达某阈值后突变出现的能力(争议) | §17.5 |
| EP | 专家并行 | MoE 专家分布到多卡,All-to-All 路由 | §14.8 |
| Few-Shot / ICL | 小样本 / 上下文学习 | prompt 中给示例,模型类比学习 | §32.3 |
| FIM | 中间填空 | 把 token 拆成 prefix/middle/suffix 重排训练 | §12.4 |
| FP8 Training | FP8 训练 | 8-bit 浮点训练,需 per-tile scale | §13.3.5 / §26.4 |
| Gradient Accumulation | 梯度累积 | 多个 micro-batch 累加再 step | §13.4 |
| Gradient Checkpointing | 梯度检查点 | 不存全部激活,反向时重算 | §03.7 / §13.5 |
| GRPO | 组相对策略优化 | 无 value model 的轻量 PPO | §16.7 / §28 |
| Inference-Aware | 推理感知 | scaling 把推理成本纳入考虑 | §17.4 |
| Label Smoothing | 标签平滑 | one-hot 软化成 (1-ε)y + ε/V | §04.7.3 |
| LoRA / QLoRA | 低秩 / 量化低秩适配 | 冻结主权重,只训低秩增量 | §15.5 |
| LR Schedule(Warmup + Cosine / WSD) | 学习率调度 | 升 → 稳 → 降的训练节奏 | §04.6 |
| MLM | 掩码语言建模 | BERT 风格,双向预测被掩 token | §12.3 |
| MTP | 多 token 预测 | 一次预测 N 个 token,辅助 loss + speculative | §12.5 / §26.5 |
| PP / TP / DP | 流水 / 张量 / 数据并行 | 三大分布式维度 | §14 |
| PPO / RLHF | 近端策略优化 / 人类反馈 RL | 经典对齐三阶段 | §16.3 / §16.5 |
| RLAIF | AI 反馈 RL | 用强模型替代人类标注偏好 | §16.8 |
| Scaling Law | 标度律 | loss = 幂律 + 不可约项 | §17 |
| SFT | 监督微调 | 用 (instruction, response) 对训 base 模型 | §15 |
| ZeRO / FSDP | 优化器状态切片 / 全切片 | 分布式显存节省 | §14.4 |

---

## A.4 推理与部署

| 英文 | 中文 | 一句话解释 | 章节 |
| --- | --- | --- | --- |
| AWQ / GPTQ | 激活感知 / OBQ 量化 | 4-bit 权重量化主流方案 | §21.8 / §22.3 |
| Beam Search | 集束搜索 | 同时维护 B 个候选,LLM 时代用得少 | §20.5 |
| Best-of-N | 多采样优选 | 并行 N 次采样,RM 或验证器选最优 | §20.5.2 |
| Chunked Prefill | 分块预填 | 长 prompt 切块,与 decode 交错 | §21.3 |
| Continuous Batching | 连续批处理 | 动态加入新请求,踢出已完成 | §18.5 |
| Decode | 解码 | 推理逐 token 输出阶段,memory-bound | §18.2.2 |
| Disaggregated Serving | 解耦服务 | Prefill / Decode 分集群部署 | §18.7 |
| Flash Attention | 闪光注意力 | IO-aware tiling,attention 砍 IO | §21.2 |
| GGUF | GGUF 格式 | llama.cpp 跨平台量化文件 | §22.3.1 |
| Greedy / Temperature / Top-k / Top-p / Min-p | 采样策略 | logits → token 的不同截断与采样方法 | §20 |
| KV Cache | KV 缓存 | 已算 K/V 持久化,decode 避免重算 | §19 |
| LM Head | 语言模型头 | 主干输出 → vocab logits 的线性 | §06.1.3 |
| Logit Bias | logit 偏置 | 对指定 token 加 / 减分数 | §20.7 |
| PagedAttention | 分页注意力 | KV cache 按页管理,vLLM 核心 | §19.4 |
| Prefill | 预填 | 推理读 prompt 一次性 forward 建 KV,算力受限 | §18.2.1 |
| Prefix Cache | 前缀缓存 | 多请求共享前缀 KV cache | §19.7 |
| Reranker | 重排器 | cross-encoder 精排候选 | §30.7 |
| Speculative Decoding | 推测解码 | draft 模型猜 + target 验,decode 加速 | §21.4 |
| Sliding Window Attention | 滑窗注意力 | 每 token 只看最近 W 个 | §21.6 |
| TTFT / TPOT | 首 token / 后续 token 延迟 | 推理两大延迟指标 | §18.8 |
| vLLM / SGLang / TGI / TRT-LLM / llama.cpp / Ollama | 推理引擎 | 各场景主流推理引擎 | §22.4 |

---

## A.5 DeepSeek 专题

| 英文 | 中文 | 一句话解释 | 章节 |
| --- | --- | --- | --- |
| Aux-Loss-Free Balancing | 无辅助 loss 负载均衡 | 用 bias 微调路由,不污染主 loss | §25.7 |
| DeepSeekMoE | DeepSeek-MoE | 细粒度 + 共享专家 + Top-k + bias 平衡 | §25 |
| Decoupled RoPE | 解耦旋转位置编码 | 把位置信号拆成独立通道,兼容 MLA | §24.4 |
| Device-Limited Routing | 设备受限路由 | token 至多激活 M 个组,控制 All-to-All | §25.8 |
| Distillation(R1 蒸馏) | R1 蒸馏 | 用 R1 reasoning 数据 SFT 小模型 | §28.7 |
| Fine-Grained Expert | 细粒度专家 | 把大专家切成更多小专家,组合空间大涨 | §25.3 |
| MLA | 多头潜变量注意力 | 低秩 KV + decoupled RoPE,KV 压 ~57× | §24 |
| R1 / R1-Zero | R1 / R1-Zero | 多阶段 / 纯 RL reasoning 模型 | §28 |
| Shared Expert | 共享专家 | 永远激活的专家承担通用知识 | §25.4 |
| Test-Time Scaling | 推理时标度 | 多想几秒 → 准确率上升的 scaling 维度 | §17.8 / §28.8 |
| V2 / V3 / V3.x | DeepSeek-V2/V3 | 第二/三代 MoE 旗舰 | §23 / §26 |

---

## A.6 应用前沿

| 英文 | 中文 | 一句话解释 | 章节 |
| --- | --- | --- | --- |
| Agent | 智能体 | LLM + 工具 + 记忆 + 循环 | §31 |
| ALMA / Any-Resolution | 任意分辨率 | 视觉切块支持高分辨率 | §29.6 |
| Browser / Computer Use | 浏览器 / 桌面操作 | Agent 操作 UI 的范式 | §31.8 |
| CLIP / SigLIP | 对比预训练视觉编码器 | 图文对齐的主流 encoder | §29.3 |
| CoT(Chain-of-Thought) | 思维链 | 让模型逐步推理再答 | §32.4 |
| Function Calling | 函数调用 | JSON schema 结构化 tool 调用 | §31.3 |
| Hybrid Retrieval | 混合检索 | BM25 + dense 融合 | §30.6 |
| HyDE | 假设文档检索 | 让 LLM 伪造答案做 query embedding | §30.8.1 |
| LLaVA | LLaVA | Encoder + MLP + LLM 的极简多模态 | §29.5.1 |
| Long-Context vs RAG | 长上下文 vs 检索 | 两种知识接入的取舍 | §30.9 |
| MCP(Model Context Protocol) | 模型上下文协议 | Anthropic 工具协议事实标准 | §31.7 |
| Multi-Agent | 多智能体 | 角色分工 / debate / 群体投票 | §31.6 |
| Planning / ToT / Reflection | 规划 / 思维树 / 反思 | Agent 进阶范式 | §31.4 |
| Prompt Injection / Jailbreak | 提示注入 / 越狱 | LLM 安全攻击与防御 | §32.9-10 |
| RAG | 检索增强生成 | 先检索后生成,接入外部知识 | §30 |
| ReAct | 推理 + 行动 | Thought-Action-Observation 循环 | §31.2 |
| Self-Consistency | 自一致性 | 多 CoT 投票 | §32.5 |
| Structured Output | 结构化输出 | grammar / JSON schema 受限解码 | §20.6 |
| ViT | Vision Transformer | 把图切 patch 当 token | §29.2 |

---

## A.7 数值精度与硬件

| 英文 | 中文 | 一句话解释 | 章节 |
| --- | --- | --- | --- |
| BF16 / FP16 / FP32 / FP8 | 各精度浮点 | 范围 vs 精度的取舍 | §01.7 / §13.3 |
| HBM | 高带宽显存 | GPU 主显存,推理 decode 瓶颈 | §21.1 |
| H100 / H200 / H800 / B100/B200 | NVIDIA GPU 型号 | LLM 训练 / 推理主力 | §13 / §22 |
| InfiniBand / NVLink | 高速互联 | 多卡 / 多节点通信带宽来源 | §14 |
| Memory-bound / Compute-bound | 内存 / 算力受限 | roofline 模型分类 | §21.1 |
| RoCE / RDMA | 网络协议 | 跨节点 KV cache 传输基础 | §18.7 / §19.8 |
| Tensor Core | 张量核心 | NVIDIA GPU 的 mixed precision 加速单元 | §13 |
| TFLOPS | 每秒万亿次浮点 | 算力度量 | §01.9 / §22.2 |
