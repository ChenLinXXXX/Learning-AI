# 33 Skills(Claude Skills / 可复用能力封装 / 提示工程的下一站)

> **章节定位**: Part VI 第 5 章,把 Prompt / Agent / Tool 三者沉淀成"可复用 Skill"的工程范式
> **预计阅读时间**: 50-70 分钟
> **难度**: ★★★☆☆
> **前置知识**: §31 Agent、§32 Prompt 工程、§22 部署、§34 MCP(并行)
> **后续依赖**: 真实业务

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [33.1 为什么需要 Skills?](#331-为什么需要-skills)
- [33.2 Skill 是什么](#332-skill-是什么)
- [33.3 Claude Skills 规范](#333-claude-skills-规范)
- [33.4 Skill vs Tool vs MCP](#334-skill-vs-tool-vs-mcp)
- [33.5 Skill 设计原则](#335-skill-设计原则)
- [33.6 触发机制:模型如何决定调 Skill](#336-触发机制模型如何决定调-skill)
- [33.7 跨厂商生态](#337-跨厂商生态)
- [33.8 与 Prompt 工程的关系](#338-与-prompt-工程的关系)
- [33.9 完整示例:写一个 Skill](#339-完整示例写一个-skill)
- [33.10 工程实践与坑](#3310-工程实践与坑)
- [🎨 33.10.A 通俗类比合集](#-3310a-通俗类比合集)
- [33.11 常见疑问](#3311-常见疑问)
- [33.12 本章小结](#3312-本章小结)
- [33.13 延伸阅读](#3313-延伸阅读)
- [33.14 实战练习](#3314-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

Prompt 是一次性沟通,Tool 是单次调用,MCP 是工具协议——**Skills 是把"任务级能力"封装成可重用、可分享、可版本化的单元**:一个 Skill = 描述 + 触发条件 + 操作指引 + 资源文件(脚本 / 模板 / 数据)。

Claude 2024 末推出官方 Skills 规范,基于"按需加载、模型自主决定何时使用、附带文件级资源"的设计哲学。本章把这一范式系统化:

- 它解决了 prompt 永远塞太满、tool 粒度太细、Agent 流程太硬编码这三大痛
- 它与 MCP / function calling / prompt 模板**共存**而不替代
- 它正成为 Agent 时代的"npm 包"

读完你能写出自己第一个 Skill,理解它在 prompt → tool → MCP → Skill 工程谱系中的定位。

---

## 学习目标

完成本章后,你能够:

- ✅ 用一句话说出 Skill 与 Tool / MCP / Prompt 模板的差别 (§33.4)
- ✅ 写出 Claude Skill 的标准目录结构 (§33.3)
- ✅ 设计一个 Skill 的 description 让模型能正确"嗅到"触发时机 (§33.6)
- ✅ 给出 Skill 适合 / 不适合的 3 类场景 (§33.5)
- ✅ 写一个自己用的 Skill 并跑通 (§33.9)

---

## 33.1 为什么需要 Skills?

### 33.1.1 Prompt 工程的三大痛

- **Prompt 太长**:把所有领域规则、模板、流程塞进 system prompt,token 爆且模型难抓重点
- **Tool 太细**:Function Calling 把每个 API 暴露,适合"原子操作"但不适合"工作流"级能力
- **Agent 框架太硬**:LangChain / CrewAI 的 chain / graph 在代码里写死,业务变了就改代码

### 33.1.2 Skills 的解法

> **把一个"任务级能力"打包成文件夹**,内含触发说明 + 操作步骤 + 资源,**让 LLM 自己判断何时用、用哪些资源**。

类比之前学过的:

| 抽象级别 | 例 | 用处 |
| --- | --- | --- |
| Token | `the` | 模型输入的最小单位 |
| Prompt | "你是医生..." | 单次沟通 |
| Tool / Function | `search_web(q)` | 单次调用 |
| **Skill** | "写学术论文 review" | 任务级能力 |
| MCP server | Github / DB / 文件系统 | 工具生态接入 |
| Agent | ReAct 循环 | 多步控制流 |

Skills 介于 prompt 与 agent 之间——**比 prompt 重(带文件),比 agent 轻(无控制流)**。

---

## 33.2 Skill 是什么

### 33.2.1 形式

一个 Skill 在物理上是**一个文件夹**,包含:

```text
skill_name/
├── SKILL.md          # 必填:描述 + 触发条件 + 操作指引
├── scripts/          # 可选:可执行脚本(.py / .sh)
├── templates/        # 可选:文档 / 代码 / prompt 模板
├── examples/         # 可选:示范输入输出
└── data/             # 可选:小规模参考资料(<< 100KB)
```

### 33.2.2 加载方式

模型在每次对话开始时,**只看到所有 Skill 的描述**(metadata),不看完整内容。

当用户的请求与某 Skill 描述匹配,模型决定"读这个 Skill"——这时才把 SKILL.md(以及它指向的其他文件)按需载入上下文。这叫 **progressive disclosure / lazy loading**。

### 33.2.3 与"插件 / 工具"的差别

- **Plugins(早期 ChatGPT 插件)**:静态加载,token 占用恒定
- **Skills**:按需加载,模型自主判断触发时机
- **Tools / Function Calling**:模型主动调函数,Skills 让模型主动读"文件包"
- **MCP**:外部工具协议,Skills 是 prompt + 资源的本地化封装

---

## 33.3 Claude Skills 规范

Anthropic 在 2024 末/2025 初推出 Skills 规范(在 Claude Desktop / API / Code 中都可用):

### 33.3.1 SKILL.md 必含字段

```markdown
---
name: pdf-summary
description: 当用户上传 PDF 并要求总结 / 提取要点时使用。能处理 < 200 页文档,输出结构化摘要 + 关键引用 + 数据点。不适合代码 PDF。
---

# PDF Summary Skill

## When to use
- 用户提供 .pdf 文件
- 用户明确请求 summarize / 提要 / extract key points

## What it does
1. 读取 PDF(用 scripts/extract.py)
2. 按章节分块
3. 套用 templates/summary.md 输出格式
4. 末尾附原文页码引用

## Resources
- scripts/extract.py:用 pdfplumber 抽正文
- templates/summary.md:三段式总结模板
- examples/:3 份示范输入输出
```

### 33.3.2 关键字段说明

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | ✅ | 唯一标识(kebab-case) |
| `description` | ✅ | **这是触发关键**——模型仅靠它判断是否激活 |
| 正文 | ✅ | 操作指引,中文 / 英文都行 |
| `scripts/` | ❌ | 可执行代码,模型自动调用前会读 |
| `templates/` | ❌ | 模板,模型可"复制并填充" |
| `examples/` | ❌ | few-shot 风格示范 |

### 33.3.3 描述写法的"嗅出原则"

description 是模型在没读完整 Skill 之前**唯一的判断依据**,要包含:

- **何时用**(when / 触发场景)
- **能做什么**(what / 输入输出)
- **不适合什么**(boundary / 反面)

不写好 description = 永远不会被触发。

---

## 33.4 Skill vs Tool vs MCP

四者并行,各打一拳:

| 维度 | Prompt 模板 | Tool / FC | MCP Server | **Skill** |
| --- | --- | --- | --- | --- |
| **粒度** | 一句指令 | 一个函数 | 一组工具 | 一项任务能力 |
| **加载** | 进 system prompt | schema 显式 | 远程 / stdio | **按需懒加载** |
| **包含** | 文本 | 代码逻辑 | 多个 tool | **文本 + 脚本 + 模板** |
| **触发** | 永远生效 | 模型主动调 | 模型主动调 | **模型嗅 description** |
| **可分享** | 复制粘贴 | 写入框架 | 启动 server | **文件夹分享** |
| **适合** | 全局风格 | 原子操作 | 外部系统 | **复杂任务流** |

**结合用法**:

```text
Skill = "如何完成 X 任务"的操作手册 + 资源
  ↳ 内部可以调用 Tool / MCP server 完成具体动作
  ↳ 内部可以用 prompt 模板规范输出格式
```

---

## 33.5 Skill 设计原则

### 33.5.1 适合做成 Skill 的

| 场景 | 原因 |
| --- | --- |
| **多步骤任务流**(写论文 review / 做合同审查) | 步骤稳定可复用 |
| **领域专属知识 + 模板**(法律 / 医疗 / 财报) | 文档量大,塞进 prompt 浪费 |
| **格式严格的输出**(行业报告 / 合同模板) | 模板文件可直接复用 |
| **个人偏好与风格**(写作助手 / 编码规范) | 跨会话一致 |

### 33.5.2 不适合做成 Skill 的

| 场景 | 原因 |
| --- | --- |
| 单条 API 调用 | 用 Tool / FC 更轻 |
| 高频实时操作(每秒) | Skill 加载有延迟 |
| 跨用户共享的数据 | Skill 是"操作手册",不是数据库;数据放 MCP |
| 安全敏感操作 | Skill 是 prompt 级别,无强约束;关键 gate 走代码 |

### 33.5.3 粒度直觉

- 太大:一个 Skill 包揽 "writing assistant" 整体 → 触发不精确
- 太小:每个 docx 操作一个 Skill → 数量爆炸,description 不可区分
- **甜点**:一个 Skill = 一个**领域 + 任务**的组合,5-50 个常用

---

## 33.6 触发机制:模型如何决定调 Skill

### 33.6.1 内部流程(简化)

```text
1. 会话初始化:加载所有 Skill 的 description(几行字 × N 个)
2. 用户发消息
3. 模型 system prompt 中已有 description 列表
4. 模型在生成时,**自主决定**是否"读 Skill X 的完整内容"
5. 读了之后,Skill 内容被加进上下文,模型继续生成
```

类似 RAG,但**模型自己当 retriever**,通过 description 文本匹配。

### 33.6.2 description 的关键策略

- **以场景而非功能描述**:"当用户上传 PDF 并要求摘要" > "PDF 处理工具"
- **包含具体动词与关键词**:让模型在用户输入中能"嗅到"
- **明确边界**:写"不适合代码 PDF",避免误触发
- **行数**:1-3 句最佳;太长反而模型记不住

### 33.6.3 多 Skill 冲突

如果有 3 个 Skill description 都相关:

- 模型一般会读最匹配的 1-2 个
- 复杂任务可能"并联" 多 Skill(各读各的)
- 写 description 时要让"作用域"互不重叠

### 33.6.4 强制 / 禁止触发

部分实现允许:

- `force_skill = "pdf-summary"`:强制激活
- `disable_skill = ["..."]`:本次会话不用
- 这相当于 RAG 的 "force_retrieve" 钩子

---

## 33.7 跨厂商生态

### 33.7.1 Anthropic Claude Skills(原生)

- Claude.ai 网页版 / Claude Desktop / Claude API
- 在 [claude.ai/skills](https://claude.ai) 上可创建 / 分享 Skills
- Claude Code 自动支持本地 `.claude/skills/` 目录

### 33.7.2 ChatGPT GPTs

OpenAI 在 2023 末推出 "GPTs",形式上接近 Skills 但:

- 整个对话用一个 GPT 配置(不像 Skills 可多个并存按需触发)
- 包含一个 system prompt + 工具 + 知识库文件
- 商店化分享

GPTs 与 Skills 思路一致,实现细节不同。

### 33.7.3 开源 / 第三方

- **Open WebUI** 支持类似 "function" / "model"
- **Aider / Cline / Continue** 用 prompt 文件夹模拟 Skills
- **AGENTS.md / CLAUDE.md** 是项目级别的"全局 Skill"
- 社区正讨论 "Skills as standard" 跨工具统一规范

### 33.7.4 与 MCP 联动

最佳实践:

- **MCP** 提供"工具与数据" → 外部能力的标准接入
- **Skills** 提供"如何用好这些能力" → 任务级操作手册
- 一个 Skill 可调多个 MCP 工具 + 多份模板

例子:

```text
"合同审查" Skill
   ├── 调 MCP server: Google Drive 拿合同
   ├── 调 MCP server: 法务知识库 RAG
   └── 用 Skill 自带 templates/risk_table.md 输出风险表
```

---

## 33.8 与 Prompt 工程的关系

### 33.8.1 Skills 不取代 Prompt 工程

Skill 内部仍要写好 prompt:

- 触发后,Skill 文本作为新 prompt 加进上下文
- Skill 里的"步骤指引 / 输出格式 / 拒答条件"全是 prompt 工程

### 33.8.2 区别

| Prompt 工程 | Skill 工程 |
| --- | --- |
| 一次对话的"剧本" | 跨会话可重用的"剧本库" |
| 在 system prompt / user message 中 | 在文件系统中 |
| 偏文本 | 文本 + 脚本 + 模板 + 示例 |
| 单一作者维护 | 可团队 / 社区维护 |

### 33.8.3 工程推荐

把"通用 + 高频"的 prompt 模板 → 收成 Skills,**逐步建立组织内的"Skill 库"**,类似前端的组件库。

---

## 33.9 完整示例:写一个 Skill

### 33.9.1 场景

你想做一个 "code-review" Skill:用户给一段 diff,自动按公司风格出 review。

### 33.9.2 文件结构

```text
code-review/
├── SKILL.md
├── scripts/
│   └── parse_diff.py
├── templates/
│   └── review.md
└── examples/
    ├── good_diff.txt
    ├── good_review.md
    └── bad_review_dont_do.md
```

### 33.9.3 SKILL.md 示例

```markdown
---
name: code-review
description: 当用户提供 git diff / pull request / patch 并请求 review / 检查 / 找问题时触发。输出按公司风格 review,含 bug / 性能 / 风格 / 测试 4 大类。不处理整库重构请求(请用 architecture-review)。
---

# Code Review Skill

## When to use
- 用户消息包含 `diff` / `patch` / `PR` / `pull request`
- 请求"帮我看下"、"review"、"代码评审"

## How to do it

1. 用 scripts/parse_diff.py 解析 diff,得到:
   - 改动文件列表
   - 行数统计
   - 修改类型(新增 / 修改 / 删除)

2. 按 templates/review.md 框架填写:
   - **🐛 Bug 风险**:具体行 + 解释
   - **⚡ 性能**:具体行 + 改进建议
   - **🎨 风格 / 规范**:符合公司 style guide(见 data/style.md)
   - **🧪 测试**:是否缺测试,缺哪些

3. 输出末尾给一个 1-10 的整体评分。

## Style requirements
- 中文
- 每条评论必须引用具体行号
- 不空泛(避免"这里可以优化")
- 严重 bug 用 🔴,小问题 ⚠️,建议 💡

## DO NOT
- 不要重写大段代码,只指出问题与方向
- 不要 review 二进制 / 自动生成的代码
```

### 33.9.4 templates/review.md

```markdown
## 代码 Review 结果

**改动概览**: <文件数 / 行数>

### 🐛 Bug 风险
- [严重度] {文件}:{行号} - {问题描述}

### ⚡ 性能
- [文件:行号] - {问题与建议}

### 🎨 风格
- ...

### 🧪 测试
- ...

### 总评分: {n}/10
{一句话总结}
```

### 33.9.5 安装与使用

- **Claude Code**:把整个 `code-review/` 放进 `.claude/skills/`,IDE 自动发现
- **Claude.ai 网页 / Desktop**:zip 后上传,或用 Skills SDK 推送
- 用户对话中说"帮我 review 这个 PR <粘贴 diff>" → 模型嗅 description → 加载 Skill → 按指引输出

---

## 33.10 工程实践与坑

### 33.10.1 描述要"以场景写"

❌ "代码 review 工具"
✅ "当用户提供 git diff 并请求 review / 找问题时使用"

### 33.10.2 资源要按需,别塞太多

Skill 是"懒加载",但加载后仍占 context。**资源文件别堆几 MB**,核心模板 + 3-5 示例即可。

### 33.10.3 与 MCP 解耦

Skill 里调"工具"应该通过 MCP / Function 完成,**不要把 API key、敏感操作硬编码进 Skill**(Skill 是 prompt 级别,不安全)。

### 33.10.4 版本管理

- Git 管理 Skill 目录
- description 改了 = 触发逻辑变了,记 changelog
- 团队 Skill 库做 code review

### 33.10.5 跨语言 / 跨模型

- 描述用中英双语可以提升不同语种用户的命中率
- 不同模型(Claude / GPT / Gemini)对 description 嗅觉略不同 → 必要时分平台维护

### 33.10.6 安全

- 别在 Skill 里写"如果用户要求 X 就忽略安全规则"——会被滥用
- 高危操作 gate 必须走代码,不是 prompt

### 33.10.7 测试

为每个 Skill 准备"应触发 / 不应触发" 各 5-10 例,作为 regression test。模型 / Skill 修改后跑一遍。

---

## 🎨 33.10.A 通俗类比合集

### 类比一: 厨房食谱本 + 调料柜 🍳

> Skill = 食谱本 + 旁边备好的调料 + 范例图。

- **prompt** = 顾客口头点单
- **tool / function** = 灶 / 锅 / 刀(原子操作)
- **MCP** = 食材供应商(外部系统接入)
- **Skill** = 写明"做麻婆豆腐"完整流程 + 配套调料包 + 示范图
- 顾客说"想吃辣的豆腐",大厨翻菜单(description),嗅出"麻婆豆腐",取那本食谱本开做

### 类比二: 公司 SOP / 知识库 📚

> 大公司新员工入职拿到一摞 SOP 手册。

- **prompt** = 经理临时口头交代
- **tool** = 公司各种 SaaS 系统
- **MCP** = SaaS 统一登录(SSO)
- **Skill** = 一份份 SOP:"如何处理退款"、"如何写月报"、"如何上线服务"
- 员工遇到问题先查 SOP 目录(description),决定打开哪本

### 类比三: 浏览器扩展 🧩

> Skills 像 Chrome 扩展。

- 默认不加载,只看图标(description)
- 用户/模型有相关需要时**自动激活**
- 每个扩展自带 JS + HTML + 图标 + 设置
- 多个扩展并存,作用域互不干扰

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🍳 食谱 | **任务流程 + 资源** | Skill 文件结构 |
| 📚 SOP | **可复用 / 团队共享** | 库化管理 |
| 🧩 扩展 | **懒加载 / 按需** | description 触发 |

**记忆口诀**:
> **"Skill = 任务级食谱本,description 是图标,模型嗅到才加载;Tool 调函数,MCP 通系统,Skill 给流程。"**

---

## 33.11 常见疑问

### Q1: Skills 和 GPTs 是不是一回事?
**A**: 思路相近(打包能力)但实现差异大:
- GPTs:**整个对话用一个 GPT**;切换 GPT = 切换 system + 工具
- Claude Skills:**多个 Skill 并存**,模型自动嗅出激活
Skills 更像扩展,GPTs 更像独立应用。

### Q2: 我已经在用 RAG 了,还需要 Skills 吗?
**A**: RAG 给"知识",Skills 给"流程"。两者互补:
- 用 Skill 决定"如何调用 RAG + 怎么处理结果"
- 用 RAG 给 Skill 提供动态数据

### Q3: Skills 是不是某种 Agent?
**A**: 不是。Skills 是"操作手册",无控制流。需要循环 / 工具调用时,**Skill 内部告诉模型"用 Tool / MCP 做"** —— Skill + Agent 框架协同。

### Q4: 多 Skill 触发顺序?
**A**: 主流实现里模型自由决定。复杂场景常见模式:
1. Skill A 处理用户请求拆解
2. Skill B 提供领域知识
3. Skill C 输出格式规范
模型会把三者内容综合到上下文。

### Q5: Skills 会被 prompt injection 攻破吗?
**A**: 会。Skills 本质仍是 prompt。同 §32.9 防御:外部内容包 untrusted 标签,高危操作走代码 gate。

---

## 33.12 本章小结

### 核心概念

```text
Skill = 一个文件夹 = 描述 + 触发条件 + 步骤指引 + 资源
   ├── 描述用于"嗅出触发"
   ├── 步骤是 prompt 工程
   ├── 资源是模板 / 脚本 / 示例
   └── 调外部能力用 Tool / MCP
```

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **抽象级别** | 任务级,介于 prompt 与 agent 之间 |
| **加载** | 懒加载,模型嗅 description |
| **触发** | description 写得好坏决定命中率 |
| **粒度** | 一个 Skill = 一个领域 × 一项任务 |
| **互补** | 与 Tool / MCP / Prompt 工程协同 |
| **安全** | Skill 是 prompt 级,不是安全边界 |

### 一句话总结

> **Skills 把"任务级能力"封装成文件夹,description 让模型自动嗅出触发时机,内部步骤是 prompt 工程、资源是模板与脚本、外部能力交给 Tool / MCP。它解决了 prompt 太长、tool 太细、agent 流程太硬的痛点,正在成为 Agent 时代的"npm 包"——把组织内的最佳实践沉淀为可重用、可分享、可版本化的工程资产。**

---

## 33.13 延伸阅读

### 官方文档

- Anthropic *"Agent Skills"* 官方文档(Claude Docs)
- *"Skills SDK"* CLI / Python 工具链
- *"AGENTS.md"* 项目级 Skill 约定(类似 README,但写给 Agent)

### 相关推荐

- OpenAI GPTs 设计文档
- *"Custom Instructions"* / *"Memory"* 各家功能比较
- Cline / Cursor 的 *"Rules" / .cursorrules* 实践

---

## 33.14 实战练习

### 练习 1 (★)

为你的日常写作 / 编程偏好写一个 `personal-style` Skill,放到 `.claude/skills/`,验证 Claude Code / Claude Desktop 能否触发。

### 练习 2 (★)

写一个 `commit-message` Skill:给定 diff,按 Conventional Commits 风格输出。准备 5 个测试 diff,验证触发与输出。

### 练习 3 (★★)

为 §30 的 RAG 系统写一个 `rag-answer` Skill:描述何时调 RAG、如何引用、不确定时怎么处理。

### 练习 4 (★★)

把 §33.9 的 `code-review` Skill 完整搭起来,在真实 PR 上跑 5 次,记录质量,迭代 SKILL.md。

### 练习 5 (★★★)

设计一套"Skill regression test":每个 Skill 对应 (should_trigger, shouldnt_trigger) 测试集,在 CI 上跑判断模型是否准确触发。

### 练习 6 (★★★)

为团队写一个 Skill 库 layout(目录 / 命名 / changelog 模板),提交 PR 流程,以及 code review checklist。

---

## 章节交叉引用

- 前置: [§31 Agent](31_Agent.md), [§32 Prompt 工程](32_Prompt工程.md), [§30 RAG](30_RAG.md)
- 后续: [§34 MCP 深入](34_MCP深入.md)(Skills 调 MCP 协同)
- 相关: [§22 本地部署](../Part_IV_推理与部署/22_本地部署实战.md)(Skill + 本地端点)

---

*下一章: §34 MCP 深入*
