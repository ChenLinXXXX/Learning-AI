# 31 Agent(ReAct / Function Calling / Planning / Memory / Multi-Agent / MCP)

> **章节定位**: Part VI 第 3 章,把 LLM 从"会聊"训成"会做事"
> **预计阅读时间**: 80-110 分钟
> **难度**: ★★★★☆
> **前置知识**: §15 SFT(工具数据)、§20 受限解码(JSON)、§28 R1 reasoning、§30 RAG
> **后续依赖**: §32 Prompt 工程

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [31.1 Agent 是什么?](#311-agent-是什么)
- [31.2 ReAct:推理 + 行动循环](#312-react推理--行动循环)
- [31.3 Function Calling / Tool Use](#313-function-calling--tool-use)
- [31.4 Planning:Plan-and-Execute / ToT / 反思](#314-planningplan-and-execute--tot--反思)
- [31.5 Memory:短期 / 长期 / 工具型记忆](#315-memory短期--长期--工具型记忆)
- [31.6 Multi-Agent 协作](#316-multi-agent-协作)
- [31.7 MCP(Model Context Protocol)](#317-mcpmodel-context-protocol)
- [31.8 Browser Use / Computer Use / Coding Agent](#318-browser-use--computer-use--coding-agent)
- [31.9 训练:Agent 数据 + RL](#319-训练agent-数据--rl)
- [31.10 评测:SWE-Bench / WebArena / AgentBench](#3110-评测swe-bench--webarena--agentbench)
- [31.11 工程坑](#3111-工程坑)
- [31.12 完整代码:60 行 ReAct Agent](#3112-完整代码60-行-react-agent)
- [🎨 31.12.A 通俗类比合集](#-3112a-通俗类比合集)
- [31.13 常见疑问](#3113-常见疑问)
- [31.14 本章小结](#3114-本章小结)
- [31.15 延伸阅读](#3115-延伸阅读)
- [31.16 实战练习](#3116-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

LLM Agent = **LLM + 工具 + 记忆 + 循环**,让模型不仅"输出文字",还能**采取行动**(查搜索、跑代码、调 API、操作浏览器/操作系统)。本章把 Agent 工程化拆开:

- **ReAct**:Thought → Action → Observation → Thought 循环范式
- **Function Calling**:结构化 tool 调用,JSON schema 约束
- **Planning**:Plan-and-Execute、Tree-of-Thoughts、Reflection
- **Memory**:短期(上下文)+ 长期(向量库 / 知识图谱)+ 工具型(NPC、Storm)
- **Multi-Agent**:角色分工 / debate / 群体智能
- **MCP**:Anthropic 提出的"工具协议"事实标准
- **Browser / Computer Use**:Claude Computer Use / OpenAI Operator / Manus 路线

读完你能从零写一个 60 行 ReAct,从而把 §28 R1 的 reasoning 能力延伸到"会动手"的实战 Agent。

---

## 学习目标

完成本章后,你能够:

- ✅ 写出 ReAct 单轮的 4 个字段(Thought / Action / Action Input / Observation) (§31.2)
- ✅ 用 JSON schema 设计一个 function calling 接口 (§31.3)
- ✅ 区分 Plan-and-Execute 与 ReAct 的差别 (§31.4)
- ✅ 给定场景设计三层 memory(short / long / tool) (§31.5)
- ✅ 用 60 行 Python 实现一个能算数 + 查 Wikipedia 的 ReAct Agent (§31.12)

---

## 31.1 Agent 是什么?

### 31.1.1 三个维度

| 维度 | 简化 LLM | 完整 Agent |
| --- | --- | --- |
| **输入** | 一段 prompt | 用户目标 + 工具 + 记忆 |
| **输出** | 一段文本 | 多步行动 + 中间产物 + 最终回答 |
| **环境交互** | 无 | 浏览器 / 操作系统 / API / 数据库 |

Agent 把 LLM 当作"决策中枢",其余配套(工具调度、循环控制、记忆)由外层框架管。

### 31.1.2 历史脉络

| 年份 | 里程碑 |
| --- | --- |
| 2022 | ReAct(Yao et al.)& Toolformer |
| 2023 | AutoGPT / BabyAGI / LangChain Agent 大火 |
| 2023.06 | OpenAI Function Calling 正式上线 |
| 2024 | Multi-Agent(CrewAI, AutoGen, MetaGPT) |
| 2024.11 | **MCP** 由 Anthropic 推出 |
| 2024.10 | **Claude Computer Use** 操控 GUI |
| 2025 | **Manus / OpenAI Operator / DeepSeek-V3 Agent** 等"通用 Agent"年 |

---

## 31.2 ReAct:推理 + 行动循环

### 31.2.1 范式

Yao et al., 2022。LLM 输出按固定格式,**交替进行 reasoning 与 action**:

```text
Thought: 我需要查 X 的最新数据
Action: search
Action Input: "X 2024 财报"
Observation: <搜索返回内容>

Thought: 现在我有了数据,需要算同比
Action: calculator
Action Input: "1.2e9 / 0.8e9 - 1"
Observation: 0.5

Thought: 我已经有答案了
Final Answer: X 同比增长 50%
```

### 31.2.2 实现

LLM prompt 包含:

- 系统提示:可用工具 + 输出格式描述
- 历史:之前所有 Thought / Action / Observation
- 当前 step:让 LLM 续写

每次 LLM 输出后,**外层解析**:

- 看到 `Action:` → 调对应工具,把结果填回 `Observation:`,继续
- 看到 `Final Answer:` → 结束循环

### 31.2.3 为什么有效

ReAct 用 in-context learning(Few-shot 示例)就能让 LLM 学会"我该 reason 还是 act"。**不需要训练**,纯 prompt 工程。

### 31.2.4 局限

- 严格依赖输出格式 → 容易"格式飞了"(解析失败)
- 长任务多轮后,context 爆炸
- 无显式 plan,容易"绕路"

→ Function Calling 与 Planning 来救场。

---

## 31.3 Function Calling / Tool Use

### 31.3.1 核心思想

不再让 LLM 输出"Action: search\nAction Input: ..."自由文本,而是**让它输出严格 JSON**:

```json
{
  "tool_calls": [
    {"name": "search", "arguments": {"query": "X 2024 财报"}}
  ]
  // ↑ 严格 schema
}
```

外层直接 parse JSON,调对应函数,把结果以 `role: tool` 注入对话。

### 31.3.2 schema 定义

```json
{
  "name": "search",
  "description": "在互联网上搜索",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "max_results": {"type": "integer", "default": 5}
    },
    "required": ["query"]
  }
}
```

LLM 看到 schema,知道何时调什么 tool、怎么填参数。

### 31.3.3 模型支持

- OpenAI: `tools` 字段 + `tool_choice`
- Anthropic: `tools` + `tool_use` content block
- Qwen / DeepSeek / Llama-3.1+:统一的 OpenAI 兼容 schema
- 大多数 7B+ 开源模型经过 SFT 后都支持

### 31.3.4 严格输出:Structured Output

近年趋势:服务端用 **constrained decoding**(§20.6)强制输出合法 JSON schema,**0% 解析失败**。OpenAI Structured Output、Anthropic Tool Use、vLLM grammar mode 都已支持。

### 31.3.5 工具粒度设计

- **粗粒度**:`search_web`、`run_code`、`read_file` 一把抓
- **细粒度**:每个 API 单独一个 tool
经验:**5-10 个工具是甜点**,> 20 时模型容易混淆,需 router agent 或 RAG-on-tools。

---

## 31.4 Planning:Plan-and-Execute / ToT / 反思

### 31.4.1 ReAct 的天花板

ReAct 是"走一步看一步",对**多步长链任务**容易绕路或忘记目标。引入显式 plan 改善。

### 31.4.2 Plan-and-Execute

```text
Phase 1 (planner): LLM 把任务拆成步骤清单
   1. 搜索 X 2024 年财报
   2. 提取关键数字
   3. 计算同比增长
   4. 总结结论

Phase 2 (executor): 按清单逐步执行,必要时回头改 plan
```

planner 与 executor 可以是同模型也可以是不同模型。

### 31.4.3 Tree of Thoughts(ToT)

Yao et al., 2023。把"推理路径"变成树:

- 每步 LLM 生成多个候选 thought
- 评估器(self-evaluate / heuristic / value model)打分
- DFS / BFS / Beam 搜索最有希望路径

适合**搜索空间明确**的任务(24 点、创意写作、规划)。

### 31.4.4 Reflection(自反思)

Shinn et al., 2023(Reflexion)。Agent 失败后,**生成对失败原因的反思,加入 memory**,下次避坑。

形式上:

```text
1. Agent 试一次任务
2. 评估结果(成功 / 失败)
3. 如果失败:LLM 写一段 reflection 解释为什么
4. 把 reflection 加进 system / memory
5. 重试,直至成功或超 max attempts
```

### 31.4.5 与 R1 reasoning 的关系

§28 的 R1 把"长思维链 + 自检"硬塞进模型本身。Planning / Reflection 把同样的能力在**外层框架**里实现。**两条路径正在合流**:**Reasoning model 内化 + Agent 框架外化**,2025+ 大量工作探索两者协同。

---

## 31.5 Memory:短期 / 长期 / 工具型记忆

### 31.5.1 三类 memory

| 类型 | 范围 | 实现 |
| --- | --- | --- |
| **Short-term(上下文)** | 当前会话 | 直接放 prompt |
| **Long-term(跨会话)** | 用户级 / 知识级 | 向量库(类 RAG)+ 摘要 |
| **Tool memory** | Agent 自己的"工作笔记" | 文件 / 数据库 / scratchpad |

### 31.5.2 上下文压缩

长对话超过 context limit 时:

- **滑窗**:只保最近 N 轮 + system
- **摘要**:LLM 周期性把旧轮总结成短摘要
- **向量化**:把所有历史 embedding 入库,需要时检索

### 31.5.3 长期记忆 = RAG over history

把 Agent 的过往对话 / 决策 / 反思全部入库,查询时按相似度回填——这就是"个性化 Agent"的关键。

代表实现:Mem0、LangChain Memory、CharacterAI 长期角色。

### 31.5.4 Storm / NPC 等模拟记忆

Park et al., 2023 的 *"Generative Agents"*:25 个 NPC 在小镇生活,各自记忆、反思、计划,涌现复杂社会行为。这套"分层记忆"思路被 CrewAI 等多 Agent 框架借鉴。

---

## 31.6 Multi-Agent 协作

### 31.6.1 基本模式

- **角色分工**:Manager / Researcher / Coder / Reviewer 各司其职
- **辩论(Debate)**:多个 Agent 对同问题提出观点 → 对辩 → 收敛
- **群体投票**:N 个 Agent 各答 → 多数票

### 31.6.2 框架

- **AutoGen**(Microsoft):支持多 Agent + 工具
- **CrewAI**:角色化、流程化
- **MetaGPT**:模拟软件公司多角色
- **LangGraph**:基于图的 Agent 编排

### 31.6.3 何时多 Agent 真有价值?

实证:**简单任务上多 Agent 不一定胜过单 Agent + ReAct**,因为:

- 通信开销大(每对 Agent 都过 LLM)
- 一致性维护难
- "群体智慧"未必发生

**真正胜场**:任务可清晰拆解、需要不同角色专长(写代码 + 评审 + 调测)。

### 31.6.4 趋势

2025+ 趋势:**单 Agent + 强大基模(R1 / V3 / GPT-4o)**常常打败多 Agent + 弱基模。多 Agent 更适合企业流程(明确 KPI / 工序),非通用问答。

---

## 31.7 MCP(Model Context Protocol)

### 31.7.1 背景

每家工具都自定义 schema、API、auth → Agent 接 100 个工具像写 100 份适配器。

### 31.7.2 MCP

Anthropic 2024.11 提出的开放协议,**让"工具 / 数据源"用统一协议向 Agent 暴露能力**:

```text
[MCP Server(工具/数据源)]
    ├ resources/    可读资源(文件、URL、数据库视图)
    ├ tools/        可调函数
    └ prompts/      预设 prompt 模板

[MCP Client(Agent/IDE)]
    ↔  通过 JSON-RPC over stdio / SSE 与 Server 通信
```

### 31.7.3 影响

- Claude Desktop、Cline、Cursor 等都已支持
- "一处实现,处处可用":Github / Slack / DB / 本地文件系统的 MCP server 数百个
- 类似 LSP(Language Server Protocol)之于编辑器——**正在成为 Agent 工具生态的事实标准**

### 31.7.4 与 OpenAI Functions 的关系

并行而非互斥:MCP 是**外部工具**协议,FC 是**单次模型调用**的格式。生产 Agent 常 MCP(发现工具)+ FC(实际调用)结合。

---

## 31.8 Browser Use / Computer Use / Coding Agent

### 31.8.1 浏览器 Agent

让 Agent 操作 web:

- 视觉路线:截图 → 多模态 LLM → 鼠标点击坐标
- DOM 路线:解析 HTML → LLM 选 element → JS 操作
- 主流框架:Playwright + Vision LLM(WebArena、Browser-Use)

### 31.8.2 Computer Use(Claude / OpenAI Operator)

更激进:**让 Agent 截图 + 鼠标键盘控制整个桌面**。

- 截图 → 视觉 LLM 决定下一步操作
- 操作粒度:click(x, y) / type / scroll / key
- 应用:文件操作、表单填写、跨应用流程

### 31.8.3 Coding Agent

让 Agent 写代码 + 跑 + 修:

- **SWE-Agent / Aider / Cursor Agent / Devin**
- 工具:read_file / write_file / run_shell / git
- 评测:SWE-Bench Verified
- 2024 末 SOTA 已能修真实 GitHub bug ~ 50%+

### 31.8.4 通用 Agent(Manus 路线)

2025 初出现的"全栈 Agent":网页 + 终端 + 文件 + API + 长期任务跑数小时。代表:Manus、Multi-On、Devin 2.0。仍处于"demo 惊艳、生产偶尔翻车"阶段。

---

## 31.9 训练:Agent 数据 + RL

### 31.9.1 SFT 数据

要 LLM 擅长 Agent 必须有相应训练数据:

- 多轮 (Thought, Action, Observation) trace
- function calling 格式数据
- 工具使用范例(代码 / API)

主流模型(LLaMA-3.1+ / Qwen-2.5+ / DeepSeek-V3 / GPT-4o / Claude)都做过专门 Agent SFT,**单纯 base 模型 ReAct 很弱**。

### 31.9.2 RL 训练 Agent

reward:任务成功(代码通过测试、操作完成网页表单等)。GRPO / RLHF / DPO 都被尝试用于 Agent。

- WebGPT、ToolFormer 早期尝试
- 2025+ Agent RL 训练系统兴起(verl、open-r1-agent 等)

### 31.9.3 蒸馏 Agent

类 R1 蒸馏:用强 Agent(GPT-4 / Claude)生成大量优质 trace → SFT 小模型。开源 7B-32B 模型 Agent 能力大幅提升。

---

## 31.10 评测:SWE-Bench / WebArena / AgentBench

| Benchmark | 任务 |
| --- | --- |
| **SWE-Bench Verified** | 修真实 GitHub bug |
| **WebArena** | 浏览器多步任务 |
| **AgentBench** | 多种 Agent 任务综合 |
| **τ-Bench** | 客服 Agent |
| **OSWorld** | 操作桌面 OS |
| **GAIA** | 通用 Agent 综合(简单 → 难) |

2024-2025 这些 benchmark 是 Agent 主战场。SOTA 持续刷新,具体数字按月变化,书中只列基准。

---

## 31.11 工程坑

| 坑 | 缓解 |
| --- | --- |
| **格式飞了** | 用 structured output / FC,而非自由文本解析 |
| **死循环 / max steps 爆** | 硬上限 + 早停启发式 |
| **工具误用 / 参数错** | 严格 schema + 类型验证 + 容错 prompt |
| **context 爆炸** | 摘要 + 长期记忆向量化 |
| **prompt injection** | 工具返回内容当数据而非指令,审计高危操作 |
| **延迟高** | 流式 + 缓存 prompt + 工具并行 |
| **成本失控** | per-task token budget + 早停 |
| **幻觉调用不存在的工具** | 工具列表显式 + reject 未知调用 |

---

## 31.12 完整代码:60 行 ReAct Agent

```python
"""
最小 ReAct Agent:支持 search + calculator 两个工具
"""
import json, re, math
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

TOOLS = {
    "search": lambda q: f"<伪造结果>关于 '{q}' 的搜索摘要…",
    "calc":   lambda expr: str(eval(expr, {"__builtins__": {}}, {"math": math})),
}

SYSTEM = """你是一个 ReAct Agent。可用工具:
- search(query): 搜索互联网
- calc(expr): 数学计算

每次只输出一种字段,严格按以下格式:
Thought: <推理>
Action: <工具名,如 search 或 calc>
Action Input: <参数,纯字符串>

或当你已得到答案时:
Thought: <最后推理>
Final Answer: <答案>"""


def parse(text):
    if "Final Answer:" in text:
        return ("final", text.split("Final Answer:")[-1].strip())
    m_act = re.search(r"Action:\s*(\w+)", text)
    m_in  = re.search(r"Action Input:\s*(.+)", text)
    if m_act and m_in:
        return ("action", m_act.group(1), m_in.group(1).strip())
    return ("noop", text)


def run(user_query: str, max_steps: int = 6):
    history = [{"role": "system", "content": SYSTEM},
               {"role": "user",   "content": user_query}]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model="local", messages=history, temperature=0.2,
            stop=["Observation:"],
        )
        text = resp.choices[0].message.content
        print(f"\n--- step {step} ---\n{text}")
        kind, *rest = parse(text)
        if kind == "final":
            return rest[0]
        if kind == "action":
            tool, arg = rest
            if tool not in TOOLS:
                obs = f"Unknown tool {tool}"
            else:
                try:
                    obs = TOOLS[tool](arg)
                except Exception as e:
                    obs = f"Tool error: {e}"
            history.append({"role": "assistant", "content": text})
            history.append({"role": "user",      "content": f"Observation: {obs}"})
        else:
            return text
    return "[max steps exceeded]"


if __name__ == "__main__":
    print(run("帮我算 (12.3 + 4.7)^2,然后告诉我结果是多少"))
```

> 真实生产用:LangGraph、CrewAI、OpenAI Agents SDK、Anthropic Agent SDK。

---

## 🎨 31.12.A 通俗类比合集

### 类比一: 实习生 + 导师 👶

> Agent = 实习生,LLM = 大脑,工具 = 公司软件。

- **ReAct** = 实习生边想边做:"我应该先查个东西" → 上网搜 → 看到结果 → 再想下一步
- **Function Calling** = 公司给实习生发**标准工单格式**,不允许写错栏
- **Plan-and-Execute** = 入职新项目,先列任务清单,再按清单逐项推进
- **Reflection** = 项目失败后写复盘,下次避坑
- **Memory** = 实习生的随身笔记本 + 公司 wiki
- **Multi-Agent** = 一个团队:PM + 工程 + 测试 + 评审,各干各的活
- **MCP** = 公司有统一 SaaS 接口,工具不必逐个适配

### 类比二: 厨房新厨师 🍳

> 单纯 LLM 是个会念食谱的人;Agent 是真上灶台炒。

- **ReAct** = 边看食谱边思考:"先备料 → 再热锅 → 再翻炒"
- **Tool** = 不同的锅、灶、刀
- **Plan-and-Execute** = 上灶前先列出 10 步,再执行
- **Reflection** = 炒糊后总结"火太大",下次小火
- **Memory** = 心里记着哪些客人爱辣 / 哪些菜常被点
- **MCP** = 餐厅设备(冰箱 / 炉灶 / 洗碗机)有统一接口,新厨师一来就能用

### 类比三: 侦探破案 🔎

> Agent 像福尔摩斯。

- **Thought** = 我推理:"嫌疑人 X 当晚不在场吗?"
- **Action** = "去现场查脚印"
- **Observation** = 脚印大小 42 码
- **Plan-and-Execute** = 先列 5 条线索清单,再逐条侦查
- **ToT** = 同时探索多种假设,每条都打分
- **Reflection** = 上次破不了案,反思自己漏看了什么
- **Multi-Agent** = 福尔摩斯 + 华生 + 雷斯垂德,角色互补
- **MCP** = 警局数据库、法医、痕迹科都用标准协议汇报

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 👶 实习生 | **学习与协作** | ReAct + Multi-Agent |
| 🍳 厨师 | **手脑结合** | Tool Use |
| 🔎 侦探 | **推理 + 行动** | Plan + Reflection + ToT |

**记忆口诀**:
> **"Thought 想 → Action 做 → Observe 看 → 循环;FC 给工具,Plan 给清单,Reflection 给复盘,MCP 给协议。"**

---

## 31.13 常见疑问

### Q1: 我用 LangChain 还是手写?
**A**: 原型 / 复杂图(LangGraph)→ 用框架。生产稳定核心 → 手写更好控制。**框架是脚手架,不是房子**。

### Q2: ReAct 的格式总解析失败怎么办?
**A**: 改用 function calling + structured output。LLM 输出严格 JSON,0% 解析失败。

### Q3: 工具数量多怎么管理?
**A**:
- 5-10 个:直接 FC 列出
- 10-50 个:按类目分组 + router agent
- 50+:tool RAG(query embedding 检索相关工具)

### Q4: Agent 跑两小时还没出结果,正常吗?
**A**: 不正常。多半:
- 死循环 → 加 max steps
- 工具 schema 不严 → LLM 反复尝试
- 任务太复杂 → 拆 sub-task,引入 plan

### Q5: 多 Agent 一定比单 Agent 好吗?
**A**: 不一定。简单任务上 overhead 大,质量未必提升。**先试单 Agent + 强 base + 好 prompt,实在不够再上多 Agent**。

---

## 31.14 本章小结

### 核心组成

```text
Agent = LLM(决策)+ Tools(动作)+ Memory(记忆)+ Loop(控制)
       └── ReAct / FC          └── 短/长/工具
       └── Plan / ToT / Reflect
       └── MCP 协议
       └── Multi-Agent 协作
```

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **ReAct** | Thought → Action → Observation 循环 |
| **FC** | JSON schema,structured output |
| **Plan** | 显式任务分解,长链不绕路 |
| **Memory** | 短期 prompt + 长期 RAG + 工具笔记 |
| **MCP** | 工具协议事实标准 |
| **Computer Use** | 视觉 + 鼠标键盘 + 桌面操作 |
| **训练** | SFT + RL + 蒸馏 |

### 一句话总结

> **Agent = LLM + 工具 + 记忆 + 循环。ReAct 给出"想-做-观"的基本骨架,Function Calling 让工具调用结构化,Plan/ToT/Reflection 把长任务变可控,Memory 三层支撑长期上下文,MCP 把工具生态标准化,Multi-Agent 把角色分工社会化——基础好的"会动手"LLM 是 2025 之后所有应用的底盘。**

---

## 31.15 延伸阅读

### 必读论文

1. Yao et al., 2022. *"ReAct: Synergizing Reasoning and Acting in Language Models"*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
2. Schick et al., 2023. *"Toolformer"*. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
3. Yao et al., 2023. *"Tree of Thoughts"*. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
4. Shinn et al., 2023. *"Reflexion"*. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
5. Park et al., 2023. *"Generative Agents"*. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

### 协议 / 框架

6. Anthropic, 2024. *"Model Context Protocol"* 官方文档
7. LangGraph / CrewAI / AutoGen / OpenAI Agents SDK 文档
8. SWE-Agent、Devin 技术博客

### 评测

9. WebArena / GAIA / AgentBench / SWE-Bench Verified leaderboard

---

## 31.16 实战练习

### 练习 1 (★)

跑通 §31.12 的 ReAct,改造让它能"先 search 后 calc"。

### 练习 2 (★★)

用 OpenAI Functions(或 vLLM tool_choice)替换文本解析,验证格式零失败。

### 练习 3 (★★)

实现 Plan-and-Execute:第一次 LLM 调用产 plan,后续按 plan 执行;对比 vs ReAct 的步数与质量。

### 练习 4 (★★★)

实现长期 memory:每完成任务把摘要 + 反思入向量库,新任务先检索相似历史;在你常用任务上做 3 天观察。

### 练习 5 (★★★)

跑一个 MCP server(本地文件系统 / SQLite 任选),用 Claude Desktop 或 Cline 连上,验证 Agent 调用工具效果。

### 练习 6 (★★★)

复现 Reflexion:给 GSM8K 错题让 Agent 失败 → 反思 → 重试,记录改善率。

---

## 章节交叉引用

- 前置: [§15 SFT](../Part_III_训练原理/15_监督微调SFT.md), [§20 受限解码](../Part_IV_推理与部署/20_采样策略.md), [§28 R1](../Part_V_DeepSeek专题/28_R1_推理路线.md), [§30 RAG](30_RAG.md)
- 后续: [§32 Prompt 工程](32_Prompt工程.md)
- 相关: [§22 本地部署](../Part_IV_推理与部署/22_本地部署实战.md)(本地 Agent 端点), [§29 多模态](29_多模态模型.md)(视觉 Agent)

---

*下一章: §32 Prompt 工程*
