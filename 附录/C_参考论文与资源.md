# 附录 C — 参考论文与资源

> **范围**: 整合本书各章引用的核心论文 + 推荐资源,**按主题归类 + arXiv 链接**。
> **用法**: 系统精读时按章选论文;面试 / 写综述时按主题速查。

---

## 目录

- [C.1 Transformer 与架构](#c1-transformer-与架构)
- [C.2 Tokenization](#c2-tokenization)
- [C.3 位置编码](#c3-位置编码)
- [C.4 归一化与残差](#c4-归一化与残差)
- [C.5 训练数据](#c5-训练数据)
- [C.6 预训练目标](#c6-预训练目标)
- [C.7 训练精度与系统](#c7-训练精度与系统)
- [C.8 分布式与并行](#c8-分布式与并行)
- [C.9 SFT / 偏好对齐](#c9-sft--偏好对齐)
- [C.10 Scaling Law](#c10-scaling-law)
- [C.11 推理优化](#c11-推理优化)
- [C.12 KV Cache / Attention 变种](#c12-kv-cache--attention-变种)
- [C.13 DeepSeek 系列](#c13-deepseek-系列)
- [C.14 多模态](#c14-多模态)
- [C.15 RAG](#c15-rag)
- [C.16 Agent](#c16-agent)
- [C.17 Prompt 工程与 reasoning](#c17-prompt-工程与-reasoning)
- [C.18 推荐资源(书 / 课程 / 博客 / 工具)](#c18-推荐资源书--课程--博客--工具)

---

## C.1 Transformer 与架构

- Vaswani et al., 2017. *Attention is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford et al., 2018-2020. GPT-1 / GPT-2 / GPT-3
- Brown et al., 2020. *Language Models are Few-Shot Learners*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Touvron et al., 2023. *LLaMA*. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Meta AI, 2024. *The Llama 3 Herd of Models*. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Devlin et al., 2018. *BERT*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

## C.2 Tokenization

- Sennrich et al., 2016. *BPE for NMT*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Kudo, 2018. *Subword Regularization*. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959)
- Kudo & Richardson, 2018. *SentencePiece*. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
- Petrov et al., 2023. *Language Model Tokenizers Introduce Unfairness*. [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)

## C.3 位置编码

- Su et al., 2021. *RoFormer / RoPE*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Press et al., 2022. *ALiBi*. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Chen et al., 2023. *Position Interpolation*. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
- Peng et al., 2023. *YaRN*. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Shaw et al., 2018. *Self-Attention with Relative Position Representations*

## C.4 归一化与残差

- He et al., 2015. *Deep Residual Learning*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Ba et al., 2016. *Layer Normalization*. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Zhang & Sennrich, 2019. *RMSNorm*. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Xiong et al., 2020. *On Layer Normalization in the Transformer*. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Wang et al., 2022. *DeepNet: 1000 Layers*. [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)
- Shazeer, 2020. *GLU Variants Improve Transformer*. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

## C.5 训练数据

- Lee et al., 2022. *Deduplicating Training Data Makes LMs Better*. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- Penedo et al., 2023. *RefinedWeb / Falcon*. [arXiv:2306.01116](https://arxiv.org/abs/2306.01116)
- Soldaini et al., 2024. *Dolma 3T-token corpus*. [arXiv:2402.00159](https://arxiv.org/abs/2402.00159)
- Gunasekar et al., 2023. *Textbooks Are All You Need*(Phi). [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)
- Wenzek et al., 2020. *CCNet*. [arXiv:1911.00359](https://arxiv.org/abs/1911.00359)
- Shumailov et al., 2023. *Model Collapse*. [arXiv:2305.17493](https://arxiv.org/abs/2305.17493)

## C.6 预训练目标

- Raffel et al., 2020. *T5: Text-to-Text Transformer*. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Tay et al., 2023. *UL2*. [arXiv:2205.05131](https://arxiv.org/abs/2205.05131)
- Bavarian et al., 2022. *Fill in the Middle*. [arXiv:2207.14255](https://arxiv.org/abs/2207.14255)
- Gloeckle et al., 2024. *Multi-token Prediction*. [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)

## C.7 训练精度与系统

- Micikevicius et al., 2017. *Mixed Precision Training*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- NVIDIA, 2022. *FP8 Formats*. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
- Chen et al., 2016. *Sublinear Memory Cost*(gradient checkpointing). [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- Korthikanti et al., 2022. *Reducing Activation Recomputation*. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
- Wortsman et al., 2022. *Model Soups*. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)

## C.8 分布式与并行

- Shoeybi et al., 2019. *Megatron-LM(TP)*. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Rajbhandari et al., 2020. *ZeRO*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Huang et al., 2019. *GPipe*. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
- Narayanan et al., 2021. *Efficient Large-Scale LM Training*. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- Lepikhin et al., 2020. *GShard*. [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
- Fedus et al., 2022. *Switch Transformer*. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

## C.9 SFT / 偏好对齐

- Ouyang et al., 2022. *InstructGPT*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Hu et al., 2021. *LoRA*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al., 2023. *QLoRA*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Zhou et al., 2023. *LIMA*. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- Wang et al., 2022. *Self-Instruct*. [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
- Christiano et al., 2017. *Deep RL from Human Preferences*. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Schulman et al., 2017. *PPO*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Rafailov et al., 2023. *DPO*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Bai et al., 2022. *Constitutional AI*. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- Lee et al., 2023. *RLAIF*. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
- Meng et al., 2024. *SimPO*. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)

## C.10 Scaling Law

- Kaplan et al., 2020. *Scaling Laws for LMs*. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al., 2022. *Chinchilla*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Wei et al., 2022. *Emergent Abilities*. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)
- Schaeffer et al., 2023. *Are Emergent Abilities a Mirage?*. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
- Muennighoff et al., 2023. *Data-Constrained Scaling*. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
- Krajewski et al., 2024. *MoE Scaling Laws*. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871)

## C.11 推理优化

- Dao et al., 2022. *FlashAttention*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, 2023. *FlashAttention-2*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Pope et al., 2023. *Efficiently Scaling Transformer Inference*. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- Kwon et al., 2023. *PagedAttention / vLLM*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Zheng et al., 2024. *SGLang / RadixAttention*. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- Leviathan et al., 2023. *Speculative Decoding*. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Cai et al., 2024. *Medusa*. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Li et al., 2024. *EAGLE*. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Xiao et al., 2024. *Streaming LLM*. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Frantar et al., 2023. *GPTQ*. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al., 2024. *AWQ*. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Hooper et al., 2024. *KVQuant*. [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- Zhong et al., 2024. *DistServe*. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

## C.12 KV Cache / Attention 变种

- Shazeer, 2019. *MQA*. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al., 2023. *GQA*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

## C.13 DeepSeek 系列

- DeepSeek-AI, 2024. *DeepSeek LLM*. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)
- Dai et al., 2024. *DeepSeekMoE*. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- Shao et al., 2024. *DeepSeekMath*(GRPO). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI, 2024. *DeepSeek-V2*. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI, 2024. *DeepSeek-V3 Technical Report*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-AI, 2025. *DeepSeek-R1*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

## C.14 多模态

- Dosovitskiy et al., 2020. *ViT*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Radford et al., 2021. *CLIP*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Zhai et al., 2023. *SigLIP*. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- Alayrac et al., 2022. *Flamingo*. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- Li et al., 2023. *BLIP-2*. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- Liu et al., 2023. *LLaVA*. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- Bai et al., 2023. *Qwen-VL*. [arXiv:2308.12966](https://arxiv.org/abs/2308.12966)
- Wang et al., 2024. *Qwen2-VL*(native-resolution)
- DeepSeek-AI, 2024. *DeepSeek-VL*

## C.15 RAG

- Lewis et al., 2020. *RAG*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin et al., 2020. *DPR*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Gao et al., 2022. *HyDE*. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Liu et al., 2023. *Lost in the Middle*. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- BAAI *BGE / BGE-M3 / BGE-Reranker* 模型卡

## C.16 Agent

- Yao et al., 2022. *ReAct*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Schick et al., 2023. *Toolformer*. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Yao et al., 2023. *Tree of Thoughts*. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- Shinn et al., 2023. *Reflexion*. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Park et al., 2023. *Generative Agents*. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- Anthropic, 2024. *Model Context Protocol* 官方文档

## C.17 Prompt 工程与 reasoning

- Wei et al., 2022. *Chain-of-Thought*. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Kojima et al., 2022. *Zero-shot CoT*. [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)
- Wang et al., 2022. *Self-Consistency*. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- Madaan et al., 2023. *Self-Refine*. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- Willard & Louf, 2023. *Outlines / Constrained Decoding*. [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)

## C.18 推荐资源(书 / 课程 / 博客 / 工具)

### 教材

- Goodfellow et al. *Deep Learning*(经典教科书)
- MacKay. *Information Theory, Inference, and Learning Algorithms*(免费 PDF)
- Strang. *Introduction to Linear Algebra*
- Trefethen & Bau. *Numerical Linear Algebra*

### 课程 / YouTube

- Karpathy. *"Neural Networks: Zero to Hero"* / *"Let's build GPT"* / *"Let's build the GPT Tokenizer"*
- 3Blue1Brown. *"Essence of Linear Algebra"* / *"But what is a GPT?"*
- Stanford CS224N / CS336

### 博客

- Lilian Weng *"LLM Powered Autonomous Agents"* / *"Prompt Engineering"* / *"The Transformer Family v2"*
- The Illustrated Transformer / GPT-2 / Stable Diffusion
- Anthropic *"Mathematical Framework for Transformer Circuits"*

### 实战工具

- **HuggingFace** transformers / datasets / accelerate / trl / peft
- **vLLM / SGLang / TensorRT-LLM / llama.cpp / Ollama / MLX**
- **LangChain / LlamaIndex / LangGraph / CrewAI / AutoGen**
- **DeepSpeed / Megatron-Core / nanotron**
- **Outlines / lm-format-enforcer**(structured output)
- **DSPy**(prompt as code)

### Leaderboards / 评测

- **LMSys ChatBot Arena**
- **Open LLM Leaderboard**(HuggingFace)
- **OpenCompass**(中文)
- **SWE-Bench Verified** / **WebArena** / **GAIA**
- **MTEB**(embedding)
- **MMBench / SEED-Bench / OCRBench**(多模态)
