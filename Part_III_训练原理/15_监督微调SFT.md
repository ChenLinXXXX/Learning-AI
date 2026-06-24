# 15 监督微调 SFT(Instruction Tuning / Chat Template / LoRA / 数据工程)

> **章节定位**: Part III 第 5 章,把"会续写"的预训练模型变成"会听话"的助手
> **预计阅读时间**: 60-90 分钟
> **难度**: ★★★☆☆
> **前置知识**: §10 Block / §12 NTP loss / §13 训练循环 / §05 Tokenization(chat template)
> **后续依赖**: §16 偏好对齐(RLHF / DPO 接力)、§22 部署、§31 Agent

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [15.1 为什么需要 SFT?](#151-为什么需要-sft)
- [15.2 数据格式:Instruction / Chat / 多轮](#152-数据格式instruction--chat--多轮)
- [15.3 Loss Masking:只算助手响应](#153-loss-masking只算助手响应)
- [15.4 数据构造:人工 / 蒸馏 / 自指令](#154-数据构造人工--蒸馏--自指令)
- [15.5 全参 SFT vs LoRA / QLoRA](#155-全参-sft-vs-lora--qlora)
- [15.6 训练超参与稳定性](#156-训练超参与稳定性)
- [15.7 多任务混合与课程](#157-多任务混合与课程)
- [15.8 灾难性遗忘与缓解](#158-灾难性遗忘与缓解)
- [15.9 评估:对齐能力指标](#159-评估对齐能力指标)
- [15.10 完整代码实现:LoRA SFT 训练](#1510-完整代码实现lora-sft-训练)
- [🎨 15.10.A 通俗类比合集](#-1510a-通俗类比合集)
- [15.11 常见疑问](#1511-常见疑问)
- [15.12 本章小结](#1512-本章小结)
- [15.13 延伸阅读](#1513-延伸阅读)
- [15.14 实战练习](#1514-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

预训练模型("base model")只会"续写",真要落地为 ChatGPT / Claude 风格的助手,必须经过**监督微调(Supervised Fine-Tuning, SFT)**。SFT 在结构上等价于 §12 的 NTP,**唯一差异是**:

- 数据从"互联网原文"换成"人类示范的 (instruction, response) 对"
- Loss 只算 **assistant 的回复部分**,不算 user / system 部分
- 数据量从 T 级别降到 1k-1M 级别

本章把 SFT 的工程细节拉通:数据格式 / loss masking / 全参 vs LoRA / QLoRA / 多任务混合 / 灾难性遗忘 / 评估。

---

## 学习目标

完成本章后,你能够:

- ✅ 区分 base / instruct / chat 三种模型阶段 (§15.1)
- ✅ 写出多轮对话样本的 loss mask 构造逻辑 (§15.3.2)
- ✅ 设计一个 LoRA SFT 配置(rank、alpha、target_modules、lr) (§15.5.3)
- ✅ 用 QLoRA 在 24 GB 单卡上微调 70B 模型 (§15.5.4)
- ✅ 估算 1 M 条 SFT 数据训练 LLaMA-7B 的算力与时间 (§15.6.4)

---

## 15.1 为什么需要 SFT?

### 15.1.1 base model 的不足

预训练模型见过整个互联网,但**没人教过它"如何回答问题"**。Prompt "Who is the president of the US?" 它可能续写"...is a question many ask online..." 而不是直接答出名字——因为续写整个网页才是它的"目标"。

### 15.1.2 SFT 的目标

教会模型理解三件事:

1. 把输入识别为"用户问题",而非"待续写的网页"
2. 给出"直接、有用、安全"的回答
3. 遵守对话格式(轮次、角色、停止符)

### 15.1.3 阶段名称

| 阶段 | 名称 | 目标 |
| --- | --- | --- |
| 1 | **base / pretrain** | 通用语言建模 |
| 2 | **SFT / instruct** | 听从指令,格式正确 |
| 3 | **RLHF / DPO** | 偏好对齐,质量提升 (§16) |
| 可选 | **continued pretraining** | 领域适配(医疗、法律…) |

---

## 15.2 数据格式:Instruction / Chat / 多轮

### 15.2.1 早期 Instruction 格式(Alpaca)

```text
Below is an instruction. Write a response.

### Instruction:
{instruction}

### Input:
{input}      # 可选

### Response:
{response}
```

适合单轮任务型场景,被 Stanford Alpaca / LIMA 等使用。

### 15.2.2 Chat Template(现代)

把多轮对话用特殊 token 包裹:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好!有什么可以帮你?<|im_end|>
<|im_start|>user
1+1 等于几?<|im_end|>
<|im_start|>assistant
等于 2。<|im_end|>
```

每个模型有自己的 chat template(LLaMA-3 / Qwen / DeepSeek 各不同)——必须**严格沿用**该模型卡里给的格式,否则模型表现显著退化。

### 15.2.3 工具调用 / Function Calling

近年扩展到 Agent 场景:

```text
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "args": {"city": "Beijing"}}
</tool_call><|im_end|>
<|im_start|>tool
{"temperature": 12, "condition": "sunny"}<|im_end|>
<|im_start|>assistant
今天北京 12 度,晴朗。<|im_end|>
```

这是 SFT 数据需要专门覆盖的"工具使用范式"(详见 §31)。

---

## 15.3 Loss Masking:只算助手响应

### 15.3.1 为什么不算 user / system 部分?

如果对所有 token 都算 NTP loss:

- 模型会学会"用户怎么问"(没必要)
- 浪费学习能力在用户输入的多样性上
- 对模型的"如何回答"信号被稀释

正确做法:**只在 assistant 的回复 token 上算 loss**,user / system / template 标签全部 mask 掉。

### 15.3.2 构造 mask

```python
def build_labels(token_ids, role_spans):
    """
    token_ids: list[int],已 tokenize 的完整序列
    role_spans: list[(start, end, role)],每段 token 的角色
    返回 labels:assistant 段保留 token id,其他段填 -100
    """
    labels = [-100] * len(token_ids)
    for start, end, role in role_spans:
        if role == "assistant":
            labels[start:end] = token_ids[start:end]
    return labels
```

PyTorch 的 `F.cross_entropy(..., ignore_index=-100)` 自动跳过这些位置。

### 15.3.3 多轮对话

每一轮 assistant 段都算 loss → 一条样本相当于多个训练信号。多轮高质数据效率显著高于单轮。

### 15.3.4 是否要 mask system?

主流做法是 mask(只学回答风格,不学 system 写法)。但有团队在做"persona-following"时反而让 system loss 参与,以学会"读懂角色设定"。

---

## 15.4 数据构造:人工 / 蒸馏 / 自指令

### 15.4.1 人工标注(最贵最贵)

- OpenAI / Anthropic 的内部数据集:几万到几十万人工标注,每条几十到几百元成本
- 优势:质量高、对齐人类偏好真实
- 限制:不可规模化,慢

### 15.4.2 蒸馏(主流开源方案)

用更强模型(GPT-4 / Claude)生成 (instruction, response) 对:

- ShareGPT(用户分享的 ChatGPT 对话)
- Alpaca / Vicuna(用 GPT-3.5/4 自动生成)
- OpenHermes / Tulu(开源混合)

注意:**商业模型用户协议可能禁止用其输出训竞争模型** → 合规风险。

### 15.4.3 自指令(Self-Instruct)

Wang et al., 2022。让模型自己生成新指令:

1. 用 175 条人工种子
2. 让模型按种子风格生成新 instruction
3. 让模型给自己生成的 instruction 写 response
4. 过滤、去重,扩展数据集

Alpaca 用此法 5 万元成本生成 5 万条数据训出 LLaMA-7B 的可对话版本——成本下降两个数量级。

### 15.4.4 数据量的甜点

- LIMA 论文(Zhou et al., 2023):**1000 条精心人工数据 ≈ 1M 条普通蒸馏数据**
- DeepSeek-R1:**80 万条高质量数据**(含 reasoning),实现 strong reasoner
- 经验:**数据质量 >> 数据数量**,千条到百万条都见效

---

## 15.5 全参 SFT vs LoRA / QLoRA

### 15.5.1 全参 SFT

更新所有 $P$ 参数。显存 ≈ 16P(§13)。

- 优点:理论上限最高
- 缺点:7B 单 80GB 已紧;70B 需多卡;每个新任务一份大 checkpoint

### 15.5.2 LoRA(Low-Rank Adaptation)

Hu et al., 2021。冻结原权重,在每个 Linear 旁加一对低秩矩阵:

$$
W' = W + B A, \quad B \in \mathbb{R}^{d\times r}, A \in \mathbb{R}^{r \times d}, r \ll d \tag{15.1}
$$

只训练 $B, A$。$r$ 典型 8-64。可训参数 ~ 0.1%-1% 原模型。

工程上常加缩放:$W' = W + (\alpha / r) BA$,$\alpha$ 与 $r$ 一起调。

### 15.5.3 LoRA 配置经验

| 项 | 推荐 |
| --- | --- |
| target_modules | $W_Q, W_K, W_V, W_O$ + FFN 全部 |
| rank $r$ | 8-32(简单任务)/ 64-128(复杂任务) |
| alpha $\alpha$ | 与 $r$ 同量级,或 $2r$ |
| LoRA dropout | 0-0.1 |
| 学习率 | $1\text{-}3\times 10^{-4}$ |

### 15.5.4 QLoRA(单卡微调大模型)

Dettmers et al., 2023:把基座模型 4-bit 量化(NF4),在量化基座上加 LoRA,前向用 dequantize → 计算 → 加 LoRA → 输出。

效果:**70B 模型可在 48 GB 单卡微调**;LLaMA-2-13B 可在 RTX 3090 24 GB 上跑。

### 15.5.5 何时选哪个?

| 场景 | 推荐 |
| --- | --- |
| 单卡微调小模型(< 13B) | LoRA |
| 单卡微调大模型(< 70B) | QLoRA |
| 多卡 + 极致质量 | 全参 SFT |
| 多 LoRA 共存(多租户) | LoRA + 动态切换 |

---

## 15.6 训练超参与稳定性

### 15.6.1 学习率

SFT lr **小于预训练**:

| 方式 | peak LR |
| --- | --- |
| 全参 SFT | $1\text{-}5\times 10^{-5}$ |
| LoRA | $1\text{-}3\times 10^{-4}$ |
| QLoRA | $2\text{-}3\times 10^{-4}$ |

### 15.6.2 Epoch 数

通常 1-3 epoch。超过 3 epoch 易过拟合,模型开始"复读"训练样本。

### 15.6.3 Batch size

- 单卡:micro-batch 1-4,gradient accumulation 达到有效 batch 32-128
- 多卡:有效 batch 128-512

### 15.6.4 算力估算

公式 $C \approx 6PD$(§01.9):

- 7B 模型 + 1M 样本(每样本 1k token = 1B token)= 6 × 7B × 1B = $4.2\times 10^{19}$ FLOPs
- 单 H100(312 TFLOPS) 大约需要 $4.2 \times 10^{19} / 3.12\times 10^{14} \approx 1.3\times 10^5$ 秒 ≈ **36 GPU 小时**
- LoRA 实际算力 ≈ 同量级(forward 仍跑完整),但显存大幅省

### 15.6.5 调度

短训练用线性 warmup(几百步)+ linear / cosine 衰减;不用复杂 WSD。

---

## 15.7 多任务混合与课程

### 15.7.1 任务类型

- **对话**:日常问答、闲聊
- **代码**:补全、解释、修 bug
- **数学 / 推理**:逐步求解、CoT
- **工具使用**:function calling、Agent
- **多语言**:翻译、跨语言对话
- **安全 / 拒绝**:不当请求的礼貌拒绝

### 15.7.2 混合比例

无银弹,但有经验:

- 通用对话 50-70%
- 代码 / 数学 15-25%
- 安全样本 5-10%
- 多语言按目标语种数据按比例

DeepSeek-V3 / LLaMA-3 都公布了详细混合配方,可作起点。

### 15.7.3 课程学习

近年趋势:**先教简单任务再教复杂任务**。例:先 short-form chat → 再 long-form reasoning → 再 tool use。

### 15.7.4 退火与高质量子集

最后几个 epoch 切到精挑细选的"金标"数据,做 final polish。

---

## 15.8 灾难性遗忘与缓解

### 15.8.1 现象

SFT 后,模型在原始 benchmark(MMLU、HumanEval)上分数下降 → "学新忘旧"。

### 15.8.2 缓解手段

| 手段 | 思路 |
| --- | --- |
| **小 lr + 少 epoch** | 减小漂移幅度 |
| **混入 base 数据** | SFT 数据中 10-20% 加预训练样本(continued pretraining 风格) |
| **LoRA** | 不动 base,自然不遗忘 |
| **EWC / Replay**(理论方案) | 大模型上几乎不用 |

实测最有效的还是前两条:**小 lr、少 epoch、混 base 数据**。

---

## 15.9 评估:对齐能力指标

### 15.9.1 自动评测

- **AlpacaEval / Arena-Hard**:相对胜率
- **MT-Bench**:多轮指令评分(GPT-4 judge)
- **HumanEval / MBPP**:代码
- **MMLU / CEval**:知识

### 15.9.2 人类评测

更可靠但更贵。盲对比 model A vs B,记录胜率。OpenAI / Anthropic 内部主要靠这个。

### 15.9.3 安全 / 拒答评测

- **HarmBench / Do-Anything-Now**:测试是否被越狱
- **WildGuard**:实际危害评测

### 15.9.4 "Refusal-overfit" 信号

SFT 安全数据过多 → 模型过度拒答(用户问"如何切洋葱不流泪"也拒)。可用专门"无害问题"测试集监控误拒率。

---

## 15.10 完整代码实现:LoRA SFT 训练

下面用 HuggingFace `transformers` + `peft` 库,演示完整 LoRA SFT 流程。

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载模型与 tokenizer
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# 2. LoRA 配置
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()    # 显示 ~0.5% 可训参数

# 3. 加载数据(假设 jsonl,每行 {"messages": [...]})
ds = load_dataset("json", data_files="sft.jsonl")["train"]

def render_and_tokenize(ex):
    text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
    enc = tokenizer(text, max_length=2048, truncation=True)
    # 构造 labels:assistant 段保留,其余 -100
    enc["labels"] = build_labels(enc["input_ids"], ex["messages"])
    return enc

ds_tok = ds.map(render_and_tokenize, remove_columns=ds.column_names)

# 4. 训练
args = TrainingArguments(
    output_dir="out", num_train_epochs=2,
    per_device_train_batch_size=2, gradient_accumulation_steps=16,
    learning_rate=2e-4, warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True, logging_steps=10, save_steps=500,
    gradient_checkpointing=True,
)
trainer = Trainer(model=model, args=args, train_dataset=ds_tok)
trainer.train()

# 5. 保存(只存 LoRA 权重,几十 MB)
model.save_pretrained("out/lora-final")
```

QLoRA 只需把第 1 步 `AutoModelForCausalLM.from_pretrained` 加 `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` 即可。

---

## 🎨 15.10.A 通俗类比合集

### 类比一: 大学生入职新公司 🏢

> 预训练 = 读完大学;SFT = 入职培训。

- **base model** = 大学毕业生,博学但不懂"公司文化"
- **SFT 数据** = 入职手册 + 师傅示范的工作流程
- **chat template** = 公司邮件 / 工单的格式规范
- **loss masking** = 培训只评估你的回答,不评估客户怎么提问
- **LoRA** = 不重学整个本科,只补一周岗位 micro-course
- **灾难性遗忘** = 入职后忘了在大学学的高数 → 偶尔复习一下基础题(混 base 数据)

### 类比二: 厨师转岗 👨‍🍳

> base model = 川菜大师,SFT = 转岗 fine dining 法餐。

- **数据格式** = 法餐厨房的术语、流程、出菜规范
- **chat template** = 各餐区交接班的标准话术
- **loss masking** = 只考核你做菜的步骤,不考核顾客点菜的语法
- **LoRA** = 不否定川菜功底,只是另外学一套法餐"调料表"
- **QLoRA** = 边缝补衣服边学新技能(基座量化)
- **多任务混合** = 同时学法、意、日料 → 全栈厨师

### 类比三: 让小孩学礼貌用语 👶

> base model = 会说话的小孩;SFT = 教他餐桌礼仪和"谢谢"。

- **示范对话** = 父母重复演示 "请—谢谢—对不起"
- **loss masking** = 不批评他听别人说什么,只批评他自己说错了什么
- **课程学习** = 先学打招呼 → 再学复杂场景对话
- **过度拒答** = 教多了"危险动作不能做",结果让他喝水也不敢喝(refusal-overfit)
- **遗忘** = 学完礼貌话忘了原来的童言童语 → 偶尔陪他自由聊天

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🏢 入职 | **角色 / 流程培训** | 数据格式 + loss masking |
| 👨‍🍳 转岗 | **专业再训练** | LoRA / QLoRA / 多任务 |
| 👶 教礼 | **行为塑造** | 灾难性遗忘 / refusal-overfit |

**记忆口诀**:
> **"SFT = 教格式 + 教语气 + 教边界;只算 assistant 的话,不算用户的提问。"**

---

## 15.11 常见疑问

### Q1: 我有 1000 条 SFT 数据,够吗?
**A**: LIMA 论文显示 1000 条高质量数据可训出可用的助手。但**质量是命门**:每条都需人工审过,涵盖多任务、多语种、多长度。不要用 1k 条都来自一个领域。

### Q2: LoRA 的 rank 怎么选?
**A**: 经验顺序:
1. 先试 $r=8$ → 是否够
2. 不够 → $r=32$,大概率够
3. 仍不够 → 全参 SFT 或换更高 $r$(64-128)

复杂任务(reasoning)需要更高 rank;chat 风格调整通常 $r=8-16$ 足够。

### Q3: SFT 后模型变笨怎么办?
**A**: 排查:
1. lr 是否过大(减半再试)
2. epoch 是否过多(< 3)
3. 数据是否单一(混入预训练样本)
4. 是否 refusal-overfit(看安全数据比例)

### Q4: chat template 一定要严格匹配吗?
**A**: 是。模型在该 template 下训练 → 推理时换格式会显著退化。`tokenizer.apply_chat_template()` 自动处理。

### Q5: SFT 之后还需要 RLHF / DPO 吗?
**A**: 看你的标准:
- 仅"能用、格式对" → SFT 够
- 追求"有用 + 安全 + 自然" → RLHF / DPO 收益明显
- 资源极有限 → DPO 性价比高(无需 reward model,详见 §16)

---

## 15.12 本章小结

### 核心流程

```text
1. 选 base 模型
2. 准备 chat-template 数据(多轮、多任务、混合)
3. 构造 labels(assistant 段保留,其余 -100)
4. LoRA / QLoRA / 全参 选型
5. 训练:bf16 + AdamW + 小 lr + 1-3 epoch
6. 评估:AlpacaEval / MT-Bench / 内部测试
```

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **SFT = 教格式 + 教任务** | base 模型不会"听话",需要 |
| **Loss masking** | 只算 assistant token |
| **LoRA / QLoRA** | 性价比首选 |
| **数据质量 > 数量** | LIMA 现象 |
| **小 lr 短 epoch** | 防遗忘 |

### 一句话总结

> **SFT 是把"会续写的 base 模型"训成"会听话的 assistant",数学上仍是 NTP loss,工程上的关键是 chat template 与 loss masking(只算 assistant);LoRA / QLoRA 是个人与小团队的性价比首选,数据质量决定上限。**

---

## 15.13 延伸阅读

### 必读论文

1. Ouyang et al., 2022. *"Training language models to follow instructions with human feedback"*. (InstructGPT,SFT + RLHF 经典)[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
2. Hu et al., 2021. *"LoRA: Low-Rank Adaptation of Large Language Models"*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. Dettmers et al., 2023. *"QLoRA: Efficient Finetuning of Quantized LLMs"*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
4. Zhou et al., 2023. *"LIMA: Less Is More for Alignment"*. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)

### 进阶论文

5. Wang et al., 2022. *"Self-Instruct: Aligning Language Models with Self-Generated Instructions"*. [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
6. Wei et al., 2022. *"Finetuned Language Models Are Zero-Shot Learners"*(FLAN)[arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
7. Tunstall et al., 2023. *"Zephyr: Direct Distillation of LM Alignment"*

### 推荐资源

- HuggingFace TRL / PEFT 文档
- DeepSpeed Chat 端到端教程
- Anthropic *"Constitutional AI"* 报告

---

## 15.14 实战练习

### 练习 1 (★)

下载 Alpaca 数据集前 5k 条,用 §15.10 的脚本 LoRA 微调 LLaMA-2-7B 或 Qwen-7B,跑 1 epoch 看效果。

### 练习 2 (★★)

实现 `build_labels`:输入 ChatML 风格的 token 序列与 role boundaries,输出 mask 后的 labels。验证 `assistant` 段 token id 保留,其他 -100。

### 练习 3 (★★)

把同一份数据分别用 LoRA r=8 与 r=64 训练,对比 MT-Bench 得分,观察"rank 收益曲线"。

### 练习 4 (★★)

用 QLoRA 在 24 GB 单卡上微调 LLaMA-2-13B(或 Qwen2.5-14B),记录显存峰值与 throughput。

### 练习 5 (★★★)

设计灾难性遗忘实验:SFT 前后分别跑 MMLU。然后在 SFT 数据中混入 20% 预训练样本,再 SFT,对比 MMLU 与对话能力的此消彼长。

### 练习 6 (★★★)

实现一个 Self-Instruct 简化版:用 GPT-4 API 生成 1k 条 (instruction, response),自动去重(MinHash),用其微调 7B 模型,评估 AlpacaEval。

---

## 章节交叉引用

- 前置: [§12 预训练目标](12_预训练目标与损失.md), [§13 训练循环](13_训练循环与混合精度.md), [§05 Tokenization](../Part_II_Transformer架构/05_Tokenization.md)
- 后续: [§16 偏好对齐](16_偏好对齐.md)
- 相关: [§22 本地部署](../Part_IV_推理与部署/22_本地部署实战.md), [§31 Agent](../Part_VI_应用前沿/31_Agent.md)(工具使用 SFT)

---

*下一章: §16 偏好对齐(RLHF / DPO / GRPO)*
