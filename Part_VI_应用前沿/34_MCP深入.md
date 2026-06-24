# 34 MCP 深入(Model Context Protocol:架构 / 协议 / 服务器与客户端 / 实战)

> **章节定位**: Part VI 第 6 章,深入 Agent 时代的"工具协议事实标准"
> **预计阅读时间**: 80-110 分钟
> **难度**: ★★★★☆
> **前置知识**: §31 Agent(已介绍 MCP 概览)、§33 Skills、基础 JSON-RPC 与进程通信
> **后续依赖**: 真实业务

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [34.1 为什么需要 MCP?](#341-为什么需要-mcp)
- [34.2 三大原语:Tools / Resources / Prompts](#342-三大原语tools--resources--prompts)
- [34.3 传输层:stdio / SSE / Streamable HTTP](#343-传输层stdio--sse--streamable-http)
- [34.4 协议层:JSON-RPC 2.0 与生命周期](#344-协议层json-rpc-20-与生命周期)
- [34.5 Server 端:能力暴露与权限](#345-server-端能力暴露与权限)
- [34.6 Client 端:Claude Desktop / Cline / Cursor / Code](#346-client-端claude-desktop--cline--cursor--code)
- [34.7 完整示例:写一个 MCP server(Python)](#347-完整示例写一个-mcp-serverpython)
- [34.8 内置 / 开源生态](#348-内置--开源生态)
- [34.9 Sampling(让 server 反向调用 LLM)](#349-samplinglet-server-反向调用-llm)
- [34.10 Roots / Elicitation:用户上下文交互](#3410-roots--elicitation用户上下文交互)
- [34.11 安全模型](#3411-安全模型)
- [34.12 与 Function Calling / Skills / Agent 的协同](#3412-与-function-calling--skills--agent-的协同)
- [34.13 工程实践与坑](#3413-工程实践与坑)
- [🎨 34.13.A 通俗类比合集](#-3413a-通俗类比合集)
- [34.14 常见疑问](#3414-常见疑问)
- [34.15 本章小结](#3415-本章小结)
- [34.16 延伸阅读](#3416-延伸阅读)
- [34.17 实战练习](#3417-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

**MCP(Model Context Protocol)** 是 Anthropic 在 2024.11 发布的开放协议,**让 LLM Agent 与外部工具 / 数据源用统一接口对话**。它定义:

- **三大原语**:Tools(可调函数)、Resources(可读资源)、Prompts(可复用模板)
- **传输层**:stdio(本地)、HTTP+SSE / Streamable HTTP(远程)
- **协议层**:JSON-RPC 2.0 + 标准初始化握手
- **进阶**:Sampling(server 反向用 LLM)、Roots(用户上下文)、Elicitation(向用户问问题)

类比:**LSP 让一个编辑器接 100 种语言;MCP 让一个 Agent 接 100 种工具**。本章从协议细节到实战 SDK,把 MCP 完整打开。

读完你能从零写一个 MCP server,部署到 Claude Desktop / Cline / Cursor / Claude Code,并理解它在生态中的位置。

---

## 学习目标

完成本章后,你能够:

- ✅ 列出 MCP 三大原语并解释各自用途 (§34.2)
- ✅ 区分 stdio 与 Streamable HTTP 适合场景 (§34.3)
- ✅ 写一个最小 Python MCP server(包含 tool + resource) (§34.7)
- ✅ 把自己的 server 配置到 Claude Desktop / Cline (§34.6)
- ✅ 解释 sampling、roots、elicitation 的反向交互机制 (§34.9-10)
- ✅ 给出 MCP 部署的 3 类安全风险与缓解 (§34.11)

---

## 34.1 为什么需要 MCP?

### 34.1.1 工具接入的 M×N 问题

没有 MCP 之前:每个 LLM 客户端(Claude Desktop / Cursor / 自己的 Agent)要接每个外部工具(GitHub / Slack / 数据库 / 文件系统),都要单独写适配器。

工具数 M × 客户端数 N = 实现成本 M×N。

### 34.1.2 LSP 的灵感

编辑器领域早就解决了同样的问题:**LSP(Language Server Protocol)**。VS Code / Neovim / Sublime 不用各自实现 100 种语言支持,语言厂商只发布一个 LSP server 即可。

MCP 把这一思想搬到 Agent 与工具:

```text
[LLM Client]  ↔ MCP  ↔  [Tool / Data Server]
```

一处实现,处处可用 → M+N 而非 M×N。

### 34.1.3 设计目标

Anthropic 公布的目标:

- **可组合**:多 server 并存,统一接口
- **可发现**:client 启动后自动列出 server 能力
- **可演进**:协议加版本,前后兼容
- **可审计**:权限明确,易做 sandboxing
- **本地优先**:默认 stdio 启动,数据不上云

---

## 34.2 三大原语:Tools / Resources / Prompts

MCP server 向 client 暴露三种能力(对应 Agent 的不同需求):

### 34.2.1 Tools(可调函数)

模型主动调用的"操作"。例:

- `search_github(query)` 搜代码
- `query_db(sql)` 跑 SQL
- `send_slack(channel, msg)` 发消息
- `create_calendar_event(...)`

每个 Tool 提供:

- 名称
- 描述
- 参数 JSON schema
- 返回结果

形式与 Function Calling 几乎一致——MCP 是其"标准协议化版本"。

### 34.2.2 Resources(可读资源)

模型可"读取"的内容,**通常不可执行,无副作用**:

- 文件:`file:///path/to/file.txt`
- 数据库视图:`db://users/123`
- HTTP URL:`https://api.example.com/...`
- 自定义 URI:`github://repo/owner/branch/path`

类比文件系统的"打开文件"——重在数据,不重在动作。

### 34.2.3 Prompts(可复用模板)

server 预定义的 prompt 模板,client 可以列出 / 调用:

- "Summarize the GitHub PR for me"
- "Generate test cases for this code"
- 内含变量,client 填充后注入对话

形式上接近 §33 Skills 的"模板"功能,但**由 server 集中维护**。

### 34.2.4 三者对照

| 原语 | 主动方 | 副作用 | 类比 |
| --- | --- | --- | --- |
| **Tools** | 模型主动调 | 可有(写 / 改 / 发) | function call |
| **Resources** | 模型 / 用户读 | 无(只读) | 文件 / URI |
| **Prompts** | 用户激活 | 无 | 模板按钮 |

一个成熟的 MCP server 通常三者都提供。

---

## 34.3 传输层:stdio / SSE / Streamable HTTP

MCP 协议本身与传输无关,常见三种载体:

### 34.3.1 stdio(本地,推荐 default)

```text
[Client 进程] ── pipe stdin/stdout ── [Server 进程]
```

- Client 启动 server 子进程,通过 stdin/stdout 收发 JSON-RPC
- **零网络**,无端口,无认证问题
- 默认场景:Claude Desktop / Cline / Code 在本机跑 server

### 34.3.2 HTTP + SSE(旧的远程方案)

- Server 起 HTTP 服务,client 通过 SSE 长连接接收 server-to-client 消息
- 用于"远程托管 server"
- 已被 Streamable HTTP 替代,新项目不要用

### 34.3.3 Streamable HTTP(2025 主流远程方案)

- 用标准 HTTP + 一个长连接 streaming endpoint
- 支持双向流式
- 更接近现代 RPC 风格

### 34.3.4 选型

| 场景 | 推荐 |
| --- | --- |
| 本地 server,只服务自己 | **stdio** |
| 团队共享 / SaaS | **Streamable HTTP** |
| Web 部署 | **Streamable HTTP** |
| 旧系统 / 兼容性 | HTTP+SSE |

---

## 34.4 协议层:JSON-RPC 2.0 与生命周期

### 34.4.1 JSON-RPC 2.0

所有消息都是 JSON 对象,三种类型:

- **Request**(带 id,期望响应)
- **Response**(对应 id 的回复)
- **Notification**(无 id,单向通知)

例:

```json
// Request
{ "jsonrpc":"2.0", "id":1, "method":"tools/list" }
// Response
{ "jsonrpc":"2.0", "id":1, "result": { "tools": [ ... ] } }
// Notification
{ "jsonrpc":"2.0", "method":"notifications/tools/list_changed" }
```

### 34.4.2 生命周期

```text
1. initialize         ─→  client 发送协议版本 + 能力
   initialized       ←─  server 返回 server 能力
2. tools/list, resources/list, prompts/list  按需查询
3. tools/call, resources/read, prompts/get   按需调用
4. notifications/...                          双向状态变化
5. shutdown(stdio)/连接关闭                   退出
```

### 34.4.3 capability negotiation

initialize 阶段双方互报"我支持什么":

- client capabilities: sampling / roots / elicitation
- server capabilities: tools / resources / prompts / logging
- 协议版本号(目前 2024-11-05、2025-03-26 等)

不支持的能力直接禁用,**保证版本演进不破坏老 client**。

---

## 34.5 Server 端:能力暴露与权限

### 34.5.1 Tool 定义

```python
# 伪 SDK 风格
@server.tool(name="weather", description="查询天气")
def weather(city: str, units: str = "celsius") -> dict:
    return fetch_weather(city, units)
```

server 启动时把所有 tool 的 schema(从签名 / docstring 推导)注册给 client。

### 34.5.2 Resource 定义

```python
@server.resource(uri="file:///workspace/{path}")
def read_workspace_file(path: str) -> str:
    return open(f"/workspace/{path}").read()
```

支持模板 URI(`{path}` 占位),client 取 URI 列表,模型按需 read。

### 34.5.3 Prompt 定义

```python
@server.prompt(name="explain_diff")
def explain_diff(diff_text: str) -> list[Message]:
    return [
        {"role": "system", "content": "你是资深 reviewer..."},
        {"role": "user",   "content": f"解释这段 diff:\n{diff_text}"},
    ]
```

### 34.5.4 权限与 sandboxing

MCP 协议本身**不强制**权限——但提供机制让 client 决定:

- 用户在 client UI **批准 / 拒绝** 每个 tool call(典型 Claude Desktop 行为)
- server 可声明 tool 是 read-only / write / destructive
- 用户可一次性允许某类操作

生产实践:**敏感 tool 默认拒,人工二次确认**。

---

## 34.6 Client 端:Claude Desktop / Cline / Cursor / Code

### 34.6.1 Claude Desktop

- 配置文件:`~/Library/Application Support/Claude/claude_desktop_config.json`(Mac)
- 添加 server:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "<YOUR_ALLOWED_PATH>"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_xxx" }
    }
  }
}
```

- 重启 Desktop,会自动 spawn 子进程,client 完成 initialize 后 tool 列表出现在 UI

### 34.6.2 Cline / Cursor / Continue

- 类似配置,把 server 注册进 IDE
- 在 chat 面板可见已加载 server 与 tools

### 34.6.3 Claude Code(CLI)

- `.mcp.json` 或全局 `~/.claude.json` 配置 server
- 启动时自动加载,工具直接出现在 Agent 可用列表

### 34.6.4 自家 Agent

任何用 Python / Node / Go 写 Agent 的项目都可作为 MCP client:

- 用官方 SDK(`@modelcontextprotocol/sdk` / `mcp` PyPI 包)
- 启动 server,收 tool list,转成你的 Agent 内部 tool

---

## 34.7 完整示例:写一个 MCP server(Python)

下面用官方 Python SDK(`pip install mcp`),做一个 weather + memo 的小 server。

```python
# server.py
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource

server = Server("demo-server")

# 假装的内存"备忘录"
MEMOS: dict[str, str] = {}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="weather",
            description="查询某城市天气(单位 celsius / fahrenheit)",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="save_memo",
            description="保存一条备忘录,key/value 字符串",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:
    if name == "weather":
        # 这里只是伪造
        return [TextContent(type="text",
            text=f"{args['city']}: 23 {args.get('units','celsius')}, sunny")]
    if name == "save_memo":
        MEMOS[args["key"]] = args["value"]
        return [TextContent(type="text", text=f"saved {args['key']}")]
    raise ValueError(f"Unknown tool {name}")


@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(uri=f"memo://{k}", name=f"Memo: {k}", mimeType="text/plain")
        for k in MEMOS
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    key = uri.replace("memo://", "")
    return MEMOS.get(key, "<not found>")


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

注册到 Claude Desktop:

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["/abs/path/server.py"]
    }
  }
}
```

重启 Desktop → 用户对话里说 "保存 memo 'todo' = 'write more skills'" → 模型识别 → 调用 `save_memo` tool → 返回 OK。

---

## 34.8 内置 / 开源生态

MCP 的真正威力来自社区生态。**官方与社区已开源数百个 server**(按热度排部分代表):

### 34.8.1 文件系统 / IDE

- `@modelcontextprotocol/server-filesystem`:本地文件读写
- `mcp-server-git`:git 操作
- `mcp-server-everything-search`:Everything 搜索
- VSCode / Neovim 的 MCP 桥

### 34.8.2 协作 / 通讯

- `mcp-server-slack`、`mcp-server-discord`、`mcp-server-gmail`
- `mcp-server-notion`、`mcp-server-linear`、`mcp-server-jira`
- `mcp-server-google-calendar`

### 34.8.3 代码 / DevOps

- `mcp-server-github`、`mcp-server-gitlab`
- `mcp-server-postgres`、`mcp-server-sqlite`、`mcp-server-mongodb`
- `mcp-server-docker`、`mcp-server-k8s`
- `mcp-server-aws` / `gcp` / `azure`

### 34.8.4 浏览器 / 抓取

- `mcp-server-puppeteer` / `browserbase`
- `mcp-server-fetch`(简化 HTTP get)

### 34.8.5 数据 / RAG

- `mcp-server-chroma` / `qdrant` / `weaviate` / `pgvector`
- `mcp-server-elasticsearch`
- `mcp-server-rag`(自定义 RAG)

### 34.8.6 LLM 工具链

- `mcp-server-openai`(server 内调 OpenAI)
- `mcp-server-anthropic`
- `mcp-server-replicate`

### 34.8.7 Meta-server

- **mcp-aggregator**:一个 server 聚合多个下游 server,降低 client 启动开销
- **mcp-proxy**:把 HTTP API 自动包装成 MCP server
- **mcp-discovery**:登记表 / 搜索 server

社区中心仓库:[github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) 与 *Awesome MCP Servers* 列表。

---

## 34.9 Sampling(让 server 反向调用 LLM)

### 34.9.1 概念

通常:client(模型)调 server 的 tool。

Sampling 反向:**server 让 client 替它调一次 LLM**,适合场景:

- Server 想"内置一个智能子任务"(归纳、改写、判断),不必自带 LLM
- 利用用户已有的 LLM 配额 / 模型选择

### 34.9.2 流程

```text
Server → Client: sampling/createMessage  (附 prompt / 参数)
Client → 用户/UI:征求是否允许采样
Client → LLM:实际调用
Client → Server:返回生成结果
```

### 34.9.3 用例

- 复杂 RAG server 内部用 LLM 做 query 改写
- 数据库 server 用 LLM 把自然语言转 SQL
- 测试 server 让 LLM 生成代码

### 34.9.4 安全

Sampling **必须**让用户批准——否则恶意 server 可以无声烧 token。client 实现要求"明确同意"。

---

## 34.10 Roots / Elicitation:用户上下文交互

### 34.10.1 Roots

Client 向 server 声明"用户授予你能访问哪些资源根":

- `file:///home/me/projects/`
- `github://my-org/`

Server 据此限定可访问范围。**默认沙盒原则**:能给的最小集合。

### 34.10.2 Elicitation

Server 向 client(进而向用户)**主动提问**:

- "你想搜哪个 repo 内的 issue?"
- "请提供你的 Slack workspace ID"

避免硬编码、避免猜参数。Elicitation 在 2025 协议版本中正式纳入。

---

## 34.11 安全模型

### 34.11.1 三类风险

| 风险 | 描述 | 缓解 |
| --- | --- | --- |
| **数据泄露** | server 读到敏感数据并外传 | sandbox、roots 限定、网络隔离 |
| **执行误用** | LLM 调用错 tool 造成数据损失 | 用户确认 / dry-run / 操作分级 |
| **Prompt Injection** | resource 内容里夹"指令"诱骗模型 | 把 resource 包 `<untrusted>`、关键操作走确认 |

### 34.11.2 沙盒建议

- Filesystem server 只允许已声明的 roots
- 数据库 server 用只读账号
- 网络 server 限制可达 host
- API key 走 env var,不写进配置共享

### 34.11.3 审计

- Client 通常打 log 每个 tool 调用 + 参数
- 生产部署加 immutable audit log
- 异常调用频率触发告警

### 34.11.4 用户教育

UI 显示"server X 想执行 Y 操作,是否允许?"——用户最终防线。

---

## 34.12 与 Function Calling / Skills / Agent 的协同

```text
[User]
  │
[Agent / Client(Claude Code, Cursor…)]
  ├── Function Calling: 调用本地实现的函数
  ├── Skills:           按需加载操作手册(§33)
  └── MCP:              统一接入外部工具 / 数据
                        ├── tools(操作)
                        ├── resources(只读数据)
                        └── prompts(模板)
```

最常见的组合:

- **Skill** 给模型"如何完成任务"的步骤指引
- 步骤里 **调 MCP tool / resource** 拿到外部能力 / 数据
- 模型用 **Function Calling 格式** 输出 tool 调用请求
- Client 把 FC 请求路由到对应 MCP server

→ **三者协同:Skill 是"做什么",FC 是"怎么调",MCP 是"由谁实现"**。

---

## 34.13 工程实践与坑

### 34.13.1 启动开销

每个 stdio server 启动需要 100ms-2s。Client 启动慢 = MCP server 太多。建议:

- 把多个相关工具合并成一个 server
- 用 mcp-aggregator 集中下游

### 34.13.2 描述质量

Tool description 直接影响模型何时调它。**描述写不好 = 工具永远不被调用**。同 §33.6 嗅出原则。

### 34.13.3 错误处理

- Tool 失败返回明确错误(stack trace 不要;一句话原因)
- 长操作支持取消 / 超时
- 模型看到错误后能 retry / 切换策略

### 34.13.4 数据量

- Resource 返回内容**不要超过模型上下文一半**(留余地给后续推理)
- 大文件 / 大表用分页 / paginate 参数

### 34.13.5 同步 vs 异步

- 长 tool(数十秒)优先返回 task_id,提供 poll 接口
- 或用 streaming(Streamable HTTP)分块返回中间结果

### 34.13.6 版本演进

- 协议有版本号,client / server 任一升级时都要 verify
- 新增 tool 没问题;删 tool 或改 schema 要发布 changelog
- 严格 backward-compatible

### 34.13.7 测试

为 MCP server 写单元测试 + 集成测试:

- 单测:每个 tool 输入输出
- 集成:用 mcp-client SDK 做完整握手 + 调用

---

## 🎨 34.13.A 通俗类比合集

### 类比一: USB 接口 🔌

> MCP 是 LLM 与外部世界的"USB"。

- **过去**:每个工具一根专用线缆,接 100 个工具要 100 种插头
- **MCP**:统一接口,工具厂商出 server,client 都能用
- **Tools** = U 盘 / 鼠标 / 键盘等可写设备
- **Resources** = 只读光盘 / 移动硬盘的只读分区
- **Prompts** = 预制的"快捷按键"
- **Sampling** = U 盘里塞了一颗小 CPU,可以反向调用电脑算力(经用户允许)

### 类比二: LSP 之于编辑器 ✏️

> 编辑器历史上的同构故事。

- 没 LSP 前:每个编辑器自己实现 Python / Rust 语法 / 自动完成
- 有 LSP 后:语言厂商出一个 server,VSCode / Vim / Sublime 都能用
- **MCP** = Agent 时代的 LSP
- **Anthropic** = 提出协议方(类比 Microsoft 提 LSP)
- **生态采纳** = 一旦 critical mass,事实标准成立

### 类比三: 餐厅总台与各档口 🍱

> 大型美食广场。

- **Client(用户的 Agent)** = 总台,客户点餐
- **MCP server** = 各档口(火锅 / 烧烤 / 寿司)
- **Tool** = 档口提供的菜
- **Resource** = 档口展示的菜单(只读)
- **Prompt** = 档口推荐的套餐(模板)
- **Sampling** = 档口想让总台帮忙广播一下信息
- **Roots / Elicitation** = 客户限定"我吃素",并主动问厨师"今天有什么蔬菜?"

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🔌 USB | **统一接口** | M+N 而非 M×N |
| ✏️ LSP | **生态标准** | 演进的预期 |
| 🍱 美食广场 | **三大原语** | Tool / Resource / Prompt 区别 |

**记忆口诀**:
> **"MCP = Agent 的 USB;三原语:Tool 动作、Resource 只读、Prompt 模板;Sampling 反向调,Roots/Elicit 限范围。"**

---

## 34.14 常见疑问

### Q1: MCP 与 OpenAI Function Calling 是替代关系吗?
**A**: 不是。FC 是单次调用的"格式",MCP 是"工具协议 + 进程模型"。最佳实践:**Agent 内部用 FC 格式,外部接 MCP server**。两者共存。

### Q2: 我已经有 LangChain Tool / LlamaIndex Tool 了,要不要迁?
**A**: 看场景:
- 工具只服务自家 Agent:留在 LangChain 也行
- 工具想跨产品复用 / 给同事用 / 给 Claude Desktop 用:**迁 MCP**

### Q3: stdio vs Streamable HTTP 怎么选?
**A**: **本地 → stdio**(简单、零网络、零认证);**远程 / SaaS → Streamable HTTP**。

### Q4: MCP 是不是只支持 Anthropic 模型?
**A**: 不是。**协议厂商无关**。OpenAI / Google / 开源模型只要客户端实现了 MCP 都能用。事实上 Cline / Cursor 配 OpenAI / 本地 vLLM 都跑 MCP。

### Q5: MCP server 多了会拖慢吗?
**A**: 会。启动开销 + 上下文中的 tool list。建议:
- 同时启用 5-15 个 server,挑常用
- 用 aggregator 合并

### Q6: 大公司能用吗?数据泄露怎么办?
**A**: 完全可以。生产做法:
- 自托管 MCP server,数据不出网
- 网络层 VPC 隔离
- 严格 roots 限定
- 审计 log 全量留档

---

## 34.15 本章小结

### 协议骨架

```text
[Client]                              [Server]
  initialize  ───────────────────────►
              ◄──── initialized
  tools/list  ───────────────────────►
              ◄──── list
  tools/call  ───────────────────────►
              ◄──── result
  resources/list, read              (类似)
  prompts/list, get                 (类似)
  notifications                     (双向)

  传输:stdio  |  Streamable HTTP
```

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **MCP** | LLM 与工具 / 数据的"USB" |
| **三原语** | Tools / Resources / Prompts |
| **传输** | stdio(本地)/ Streamable HTTP(远程) |
| **生态** | 数百开源 server + 主流 client |
| **进阶** | Sampling / Roots / Elicitation |
| **安全** | 用户确认 + sandbox + roots + audit |

### 一句话总结

> **MCP 是 Agent 时代的"USB":让 LLM 与外部工具 / 数据用 JSON-RPC + 三原语(Tools / Resources / Prompts)的统一协议对话,本地用 stdio 远程用 Streamable HTTP;它把"M×N 个适配器"压成"M+N 个 server",并通过开源生态(从文件系统、GitHub、Slack 到数据库、浏览器)快速成长为 Agent 工具生态的事实标准——是 Skills(操作手册)与 Function Calling(调用格式)三件套中"由谁实现外部能力"的那一环。**

---

## 34.16 延伸阅读

### 官方

- *"Model Context Protocol"* 官方文档与 spec(modelcontextprotocol.io)
- SDK 仓库:`@modelcontextprotocol/sdk`(TypeScript)、`mcp`(Python)
- 官方 servers 集合:[github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)

### 协议

- 当前版本 spec(`2024-11-05` / `2025-03-26` ...)
- JSON-RPC 2.0 spec
- 与 LSP / DAP 对比的设计笔记

### 推荐资源

- *"Awesome MCP Servers"* 仓库
- Anthropic Engineering blog 关于 MCP 的发布与演进
- Cline / Cursor / Claude Desktop / Claude Code 的 MCP 配置指南

---

## 34.17 实战练习

### 练习 1 (★)

把 §34.7 的 demo server 跑通:在 Claude Desktop 配置 + 重启 + 验证 "save_memo" tool 可用。

### 练习 2 (★)

为你常用的某个 API(天气 / 股票 / 翻译)写一个真实 MCP server,提供 1-2 个 tool + 1 个 resource。

### 练习 3 (★★)

用 mcp-server-filesystem + mcp-server-github + 你自己的 server,搭一个"代码 review 工作流":读本地代码 → 拉 PR → 调用 review tool。

### 练习 4 (★★)

实现一个 MCP server 加 Sampling:server 内部用 client 的 LLM 把用户的中文输入翻成英文 SQL。

### 练习 5 (★★★)

把 §33 写的 `code-review` Skill 与 MCP server 联动:Skill 描述步骤,具体读 PR 和写 comment 通过 GitHub MCP server。

### 练习 6 (★★★)

设计一份"组织内部 MCP server 治理方案":
- 命名规范、版本、文档
- 用户授权与权限分级
- 部署(本地 / VPC)与监控
- 故障恢复

---

## 章节交叉引用

- 前置: [§31 Agent](31_Agent.md)(MCP 概览), [§33 Skills](33_Skills.md), [§32 Prompt 工程](32_Prompt工程.md)
- 后续: 真实业务
- 相关: [§22 本地部署](../Part_IV_推理与部署/22_本地部署实战.md)(本地 LLM endpoint + MCP 组合), [§30 RAG](30_RAG.md)(MCP-based RAG)

---

*Part VI 至此结束。本书新增章节全部完成。*
