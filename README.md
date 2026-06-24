# Learning-AI

> 一份系统化、可复习、面向深入学习的大语言模型(LLM)知识体系
> 从数学基础到 Transformer,从训练到推理,从 DeepSeek-V3 到前沿应用

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-WIP-yellow">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Chapters" src="https://img.shields.io/badge/chapters-34-brightgreen">
  <img alt="Appendices" src="https://img.shields.io/badge/appendices-4-blueviolet">
  <img alt="Language" src="https://img.shields.io/badge/lang-zh--CN-red">
</p>

---

## ✨ 这份笔记是什么

一份**论文化、教材级、工程师视角**的 LLM 学习笔记。每章按统一固定结构组织,既可顺序学习,也可按需检索。

- 📐 **数学严谨** — 每个核心公式都有完整推导
- 🎨 **直觉先行** — 每章都有生活化类比,把抽象概念落地
- 💻 **代码可跑** — 关键算法配可运行的 PyTorch 实现
- 🧭 **路径清晰** — 4 条针对不同背景的学习路径
- 🔗 **交叉引用** — 章节间互相串联,形成知识图谱
- 📚 **论文索引** — 每章末尾附 arXiv 链接的延伸阅读

---

## 📖 目录

- [📂 仓库结构](#-仓库结构)
- [🗺 学习路径](#-学习路径)
- [📌 主题速查](#-主题速查)
- [🧱 每章固定结构](#-每章固定结构)
- [🛠 命名规范](#-命名规范)
- [🤝 贡献](#-贡献)
- [📄 License](#-license)

---

## 📂 仓库结构

```text
Learning-AI/
├── README.md                  # 总览(本文件)
├── 00_前言与学习路径.md        # 入口:摘要 + 设计哲学 + 学习路径
│
├── Part_I_数学与基础/          # 4 章 — 钢筋水泥(地基)
├── Part_II_Transformer架构/    # 6 章 — 模型骨架
├── Part_III_训练原理/          # 7 章 — 从数据到参数
├── Part_IV_推理与部署/         # 5 章 — 从输入到输出
├── Part_V_DeepSeek专题/        # 6 章 — 工程范例
├── Part_VI_应用前沿/           # 6 章 — RAG / Agent / 多模态
│
└── 附录/                       # 名词字典 / 公式速查 / 论文清单 / 路径建议
```

### 6 大部分概览

| Part | 主题 | 章节数 | 难度 |
| --- | --- | --- | --- |
| [I — 数学与基础](Part_I_数学与基础/) | 线性代数 / 概率 / 反向传播 / 优化器 | 4 | ★★☆ |
| [II — Transformer 架构](Part_II_Transformer架构/) | Tokenization / Embedding / Attention / FFN / Norm / Block | 6 | ★★★ |
| [III — 训练原理](Part_III_训练原理/) | 数据 / 预训练 / 分布式 / SFT / RLHF / Scaling Law | 7 | ★★★★ |
| [IV — 推理与部署](Part_IV_推理与部署/) | Prefill·Decode / KV Cache / 采样 / 量化 / 本地部署 | 5 | ★★★ |
| [V — DeepSeek 专题](Part_V_DeepSeek专题/) | MLA / DeepSeekMoE / V3 训练 / vs LLaMA-3 / R1 | 6 | ★★★★ |
| [VI — 应用前沿](Part_VI_应用前沿/) | 多模态 / RAG / Agent / Prompt / Skills / MCP | 6 | ★★★ |

### 附录

| 附录 | 主题 |
| --- | --- |
| [A. 名词字典](附录/A_名词字典.md) | 中英对照,200+ 术语 |
| [B. 公式速查](附录/B_公式速查.md) | 全部核心公式,便于面试冲刺 |
| [C. 参考论文与资源](附录/C_参考论文与资源.md) | 含 arXiv 链接 |
| [D. 学习路径建议](附录/D_学习路径建议.md) | 4 种背景对应 4 条路径 |

---

## 🗺 学习路径

### 1️⃣ 系统入门(零基础到工程级)

```text
00 前言 → Part I → Part II → Part III → Part IV → Part V → Part VI
```

⏱ 预计 80-100 小时

### 2️⃣ 工程师快速通道(已有 ML 基础)

```text
00 前言 → 跳过 Part I
       → Part II (重点 07/08/10) → Part III (13/14)
       → Part IV (18/19/21) → Part V → Part VI
```

⏱ 预计 40-50 小时

### 3️⃣ 面试与原理深化

```text
07 Self-Attention → 06 位置编码 → 14 分布式
→ 16 RLHF/GRPO → 19 KV Cache → 24 MLA → 25 MoE
```

⏱ 预计 20-30 小时

### 4️⃣ 部署优先(实战导向)

```text
22 本地部署 → 21 推理优化 → 20 采样 → 19 KV Cache
→ 18 推理流程 → 30 RAG → 31 Agent
```

⏱ 预计 15-20 小时

---

## 📌 主题速查

| 想了解 | 看哪里 |
| --- | --- |
| 模型怎么"懂"文字? | §05, §06 |
| Attention 数学推导 | §07 |
| 为什么除以 √d_k | §07.3 |
| RoPE 原理 | §06.4 |
| FFN 为什么是 4d | §08.2 |
| SwiGLU 公式 | §08.3 |
| MoE 路由器 | §25.3 |
| KV Cache 怎么工作 | §19 |
| RLHF / DPO / GRPO 对比 | §16 |
| Scaling Law | §17 |
| MLA 怎么压缩 KV | §24 |
| V3 为什么比 LLaMA-3 便宜 | §27 |
| 量化(INT4 / INT8 / FP8) | §21.4 |
| 如何本地跑 DeepSeek | §22 |
| RAG 系统怎么搭 | §30 |
| 怎么做 Agent | §31 |

> 📚 按论文查找 → [附录 C](附录/C_参考论文与资源.md)
> 🔤 按术语查找 → [附录 A](附录/A_名词字典.md)
> ⚡ 公式速查 → [附录 B](附录/B_公式速查.md)

---

## 🧱 每章固定结构

为便于复习,每章都包含以下固定区块:

1. **章节定位元数据** — 阅读时间、难度、前置 / 后续依赖
2. **摘要 + 学习目标**
3. **目录**(自动跳转)
4. **正文** — 直觉 → 公式 → 代码 → 复杂度
5. **🎨 通俗类比合集** — 用日常场景类比抽象概念
6. **常见疑问 Q&A**
7. **本章小结 + 一句话总结**
8. **延伸阅读** — 论文与资源
9. **实战练习**(分级 ★~★★★)
10. **章节交叉引用**

公式编号规则:`(章号.序号)`,例如 `(7.3)` 表示第 7 章第 3 个公式。

---

## 🛠 命名规范

- **Part 目录**:`Part_<罗马数字>_<中文主题>/`
- **章节文件**:`<两位数序号>_<英文标题>_<中文说明>.md`
- **附录文件**:`<字母>_<中文说明>.md`
- 全部使用**下划线 `_`** 而非连字符 `-`(避免中文路径解析问题)

示例:

```text
Part_II_Transformer架构/07_Self_Attention机制.md
附录/A_名词字典.md
```

---

## 🤝 贡献

欢迎通过 Issue 提出问题、修正错误或补充建议。如有 PR 意愿:

1. Fork → 创建分支 `feat/<主题>` 或 `fix/<主题>`
2. 遵循 [每章固定结构](#-每章固定结构) 与 [命名规范](#-命名规范)
3. 提交前用 markdownlint 检查格式
4. 提 PR 并描述变更点

### 写作约定

- 中文行文使用半角标点 + 双引号 `"..."`
- 公式用 LaTeX,块级 `$$ ... $$`,行内 `$ ... $`
- 代码块标语言(`python` / `bash` / `json` / `text`)
- 论文引用格式:`作者 et al., 年份. *"标题"*. [arXiv:XXXX.XXXXX](url)`

---

## 📄 License

本仓库内容采用 [MIT License](LICENSE) 开源。

引用请注明来源:

```text
@misc{learning-ai-notes,
  title  = {Learning-AI: A Systematic Notebook on Large Language Models},
  year   = {2026},
  url    = {https://github.com/<your-username>/Learning-AI}
}
```

---

<p align="center">
  <sub>用 ❤️ 与 ☕ 构建于 2026</sub>
</p>
