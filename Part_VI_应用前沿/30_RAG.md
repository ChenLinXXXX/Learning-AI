# 30 RAG(检索增强生成:Embedding / 向量库 / 重排 / Hybrid / Long-Context 之争)

> **章节定位**: Part VI 第 2 章,把 LLM 与"外部知识"接好
> **预计阅读时间**: 70-100 分钟
> **难度**: ★★★☆☆
> **前置知识**: §01 向量内积、§05 Tokenization、§15 SFT、§29 多模态(可选)
> **后续依赖**: §31 Agent(检索作 tool)、§32 Prompt(RAG prompt 模板)

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [30.1 为什么要 RAG?](#301-为什么要-rag)
- [30.2 经典 RAG 流水线](#302-经典-rag-流水线)
- [30.3 分块策略(Chunking)](#303-分块策略chunking)
- [30.4 Embedding 模型](#304-embedding-模型)
- [30.5 向量数据库与索引](#305-向量数据库与索引)
- [30.6 检索:稀疏 / 稠密 / Hybrid](#306-检索稀疏--稠密--hybrid)
- [30.7 重排(Reranker)](#307-重排reranker)
- [30.8 高级技巧:HyDE / Query 重写 / 多查询 / 父子文档](#308-高级技巧hyde--query-重写--多查询--父子文档)
- [30.9 Long-Context vs RAG 之争](#309-long-context-vs-rag-之争)
- [30.10 评测:命中率 / 忠实度 / 端到端](#3010-评测命中率--忠实度--端到端)
- [30.11 完整代码:最小 RAG](#3011-完整代码最小-rag)
- [🎨 30.11.A 通俗类比合集](#-3011a-通俗类比合集)
- [30.12 常见疑问](#3012-常见疑问)
- [30.13 本章小结](#3013-本章小结)
- [30.14 延伸阅读](#3014-延伸阅读)
- [30.15 实战练习](#3015-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

LLM 训练完后**知识冻结**——既不知道最新事件,也不知道你的私有文档。**RAG(Retrieval-Augmented Generation)** 用"先检索后生成"的方式让 LLM 接入外部知识:

- **离线**:把文档切块 → embedding → 入库
- **在线**:对用户 query 做 embedding → 检索 Top-k → 拼进 prompt → LLM 生成
- **进阶**:重排、Hybrid 检索、HyDE、父子文档…

本章按工程化路径展开 RAG 全栈,并讨论"长上下文 LLM 是否会取代 RAG"——结论:**两者互补,且 RAG 在多数生产场景仍占优**(成本、可控、隐私、新鲜度)。

---

## 学习目标

完成本章后,你能够:

- ✅ 画出端到端 RAG 流水线 (§30.2)
- ✅ 比较 BM25 / dense / hybrid 检索的强弱场景 (§30.6)
- ✅ 解释 reranker 为何能大幅提升 top-1 命中 (§30.7)
- ✅ 区分 chunk size 选大与选小的代价 (§30.3.3)
- ✅ 给出"何时该 RAG / 何时该长上下文"的判据 (§30.9)

---

## 30.1 为什么要 RAG?

### 30.1.1 LLM 单打独斗的四大短板

1. **知识冻结**:训练截止后的事件不知
2. **私域知识缺失**:你的产品文档、内部知识库
3. **幻觉**:对模糊问题"自信编造"
4. **上下文长度限制**:即使 128k,也装不下一个公司的全部文档

### 30.1.2 RAG 的承诺

- ✅ **更新便宜**:换文档就够,不动模型
- ✅ **来源可追溯**:回答附带 citation
- ✅ **私有可控**:数据不进训练
- ✅ **小模型也能强**:7B + 好检索,可超 70B + 无检索

但:

- ❌ 检索失败 = 生成失败,**质量上限由检索决定**
- ❌ 工程复杂度高
- ❌ 难处理"跨文档推理"

---

## 30.2 经典 RAG 流水线

```text
[离线阶段]
原始文档(PDF/HTML/MD/数据库)
   │
   ▼
[Cleaning] 去 HTML / 去 boilerplate / OCR / 表格抽取
   │
   ▼
[Chunking] 切成 200-1000 token 的段(可重叠)
   │
   ▼
[Embedding] 每段 → 向量 v ∈ R^d
   │
   ▼
[Vector DB] FAISS / Milvus / Qdrant / pgvector …

[在线阶段]
User Query
   │
   ▼
[Query Embedding] q ∈ R^d
   │
   ▼
[Retrieve Top-k] 在向量库中找 cosine 最近 k 个 chunk
   │
   ▼
[Rerank(可选)] 用 cross-encoder 重新排序,留 top-n
   │
   ▼
[Build Prompt] system + retrieved chunks + user query
   │
   ▼
[LLM 生成] 答案 + (可选)citation
```

---

## 30.3 分块策略(Chunking)

### 30.3.1 三种基本切法

| 方法 | 思路 | 适合 |
| --- | --- | --- |
| **固定长度** | 每 N token / 字符切,常带 10-20% 重叠 | 通用 baseline |
| **结构感知** | 按 markdown 标题 / HTML 块 / PDF 段切 | 文档化数据 |
| **语义切分** | 用 embedding 相似度找"语义断点" | 高质量场景 |

### 30.3.2 重叠(overlap)

每相邻 chunk 共享 50-200 token,避免在句子中间硬切丢上下文。

### 30.3.3 chunk size 取舍

- **太大(>1000 token)**:embedding 把信息"平均化",检索召回差
- **太小(<200 token)**:检索精度高但拼 prompt 时碎片多,LLM 难整合
- **甜点**:300-800 token,大多数生产场景

### 30.3.4 父子结构(Parent-Child / Hierarchical)

- 子 chunk(小,~300 token)用于检索
- 父 chunk(大,~2000 token)用于喂 LLM
- 检索命中子 → 取父喂 LLM
优点:检索精 + 上下文足。LangChain / LlamaIndex 都支持。

---

## 30.4 Embedding 模型

### 30.4.1 什么是 embedding 模型

把一段文本映射成固定维度向量,**语义近的向量近**(cosine / L2)。

通常是一个 BERT/CLM 风格编码器,**最后一层 mean pooling 或 CLS**:

$$
v = \frac{1}{T}\sum_t h_t \in \mathbb{R}^d \tag{30.1}
$$

### 30.4.2 主流模型

| 模型 | 维度 | 上下文 | 特点 |
| --- | --- | --- | --- |
| OpenAI `text-embedding-3-large` | 3072 | 8k | 商业,通用强 |
| Cohere `embed-multilingual-v3` | 1024 | 512 | 多语言 |
| BAAI **bge-m3** | 1024 | 8k | 开源 SOTA,多语言 |
| **bge-large-zh** / **gte-large-zh** | 1024 | 512 | 中文专强 |
| `jina-embeddings-v3` | 1024 | 8k | 开源多语言 |
| `nomic-embed-text-v1.5` | 768 | 8k | 完全开源 |

经验:**bge-m3 / bge-large-zh-v1.5** 是中文 RAG 的当前默认。

### 30.4.3 训练:对比 + 监督

- 大规模对比学习(InfoNCE)
- 加 supervised pair(query–positive doc + hard negatives)
- 多任务:retrieval / similarity / clustering

### 30.4.4 维度与精度

更高维度 ~ 略高质量,但**内存与检索成本线性增长**。生产权衡:

- 通用:512-1024 维
- 极致:1536-3072 维(用得起就用)
- 注意 PCA / Matryoshka 截断(部分模型支持降维)

---

## 30.5 向量数据库与索引

### 30.5.1 主流选择

| 类型 | 例子 | 特点 |
| --- | --- | --- |
| 库(嵌入应用) | **FAISS**, ScaNN, hnswlib | 单机,极快 |
| 服务(独立) | **Milvus**, **Qdrant**, **Weaviate** | 分布式,多副本 |
| 数据库扩展 | **pgvector**(Postgres), Elasticsearch dense_vector | 与现有数据库集成 |
| 云托管 | Pinecone, Vespa, MongoDB Atlas Vector | 全托管 |

### 30.5.2 ANN 索引(Approximate Nearest Neighbor)

精确搜索 $O(N)$,大数据集不可行 → 近似搜索:

- **HNSW**(Hierarchical Navigable Small World):图索引,查询 $O(\log N)$,主流默认
- **IVF**(Inverted File):聚类 + 桶,适合极大规模
- **PQ**(Product Quantization):量化压缩向量

参数权衡:**召回率 vs 速度 vs 内存**。HNSW 的 `M / ef_construction / ef_search` 是核心 3 旋钮。

### 30.5.3 元数据过滤

绝大多数生产场景需要"按租户 / 时间 / 文档类型"过滤,**先过滤再检索**(pre-filter)或**先检索再过滤**(post-filter),策略影响性能。

---

## 30.6 检索:稀疏 / 稠密 / Hybrid

### 30.6.1 稀疏检索(BM25 / TF-IDF)

经典 IR,基于关键词频:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)(k_1+1)}{f(t,d) + k_1\big(1-b+b\,\tfrac{|d|}{\bar L}\big)}
$$

特点:**关键词命中强,语义弱**。低频精确名词、产品型号、专有名词 BM25 几乎不可替代。

### 30.6.2 稠密检索(Embedding + cosine)

特点:**语义近义词 / 同义表达强,长尾关键词弱**。Embedding 模型可能"想多了"——只懂语义,不识固定字符串。

### 30.6.3 Hybrid:取长补短

把两条线分数融合:

$$
\text{score}(q,d) = \alpha \cdot \text{BM25}(q,d) + (1-\alpha) \cdot \text{cosine}(q,d) \tag{30.2}
$$

或用 **RRF(Reciprocal Rank Fusion)**:

$$
\text{RRF}(d) = \sum_i \frac{1}{k + r_i(d)}
$$

实测:Hybrid 比单条线高 10-20% recall,**几乎是生产 RAG 的默认配置**(Elasticsearch、Qdrant、Milvus 均原生支持)。

---

## 30.7 重排(Reranker)

### 30.7.1 动机

embedding 检索是 bi-encoder(query 与 doc 分别编码再 cosine),效率高但精度有限。**Cross-encoder reranker** 把 query 与 doc 拼成 `[CLS] q [SEP] d` 一起编码,**对单对计算分数**,准确得多。

### 30.7.2 经典 reranker

- BAAI **bge-reranker-v2-m3**(开源 SOTA,多语言)
- **mxbai-rerank-large**(开源)
- Cohere Rerank(商业)
- Jina Reranker

### 30.7.3 流水线

```text
向量检索 Top-100  →  Reranker 打分  →  保留 Top-5  →  喂 LLM
```

- 第一阶段(检索)粗筛广撒网
- 第二阶段(rerank)精筛少而准

### 30.7.4 收益

实测在 RAG 评测(MTEB / BEIR)上,加 rerank 把 nDCG@10 提升 5-15 个点;LLM 最终回答正确率也显著上升。

代价:rerank 是 cross-encoder,**每 query 跑 100 次 forward**——延迟与算力都增加。生产上常 Top-100 → Top-20。

---

## 30.8 高级技巧:HyDE / Query 重写 / 多查询 / 父子文档

### 30.8.1 HyDE(Hypothetical Document Embeddings)

直接用 query 做 embedding 会"问题与答案语义距离远"(query 短,doc 长)。

HyDE:**让 LLM 先伪造一段"假设答案",用它的 embedding 检索**。"想象一份理想 doc"比"查询字面"语义更近。

```text
Q = "司马懿为什么发动高平陵之变?"
→ LLM 写一段"伪答案":"司马懿在曹爽弱化士族…的背景下…"
→ 这段伪答案 embedding 做检索
```

### 30.8.2 Query 重写 / Multi-Query

- **重写**:LLM 把口语化 query 改写成关键词版
- **多查询**:LLM 生成 3-5 个不同角度 query,并行检索后合并

### 30.8.3 父子文档检索

(已在 §30.3.4 提到)。

### 30.8.4 Graph RAG / 结构化 RAG

把知识抽成知识图谱(实体 + 关系),检索路径而非文本块。微软 GraphRAG、LangChain 等都有原型。适合"跨文档实体关系"的查询。

---

## 30.9 Long-Context vs RAG 之争

### 30.9.1 问题

LLM 上下文 4k → 32k → 128k → 1M(Gemini 1.5 Pro / Qwen-2.5-1M)。 既然能塞下整本书,**还要 RAG 吗?**

### 30.9.2 Long-Context 的优势

- 工程简单(无需向量库 / 检索)
- 跨段推理友好
- 一次性看全文

### 30.9.3 Long-Context 的劣势

- **成本**:128k token prompt prefill 算力 ~ 30× 长 4k prompt
- **延迟**:TTFT 显著上升
- **稀释**:Lost-in-the-middle —— 太长时模型抓不住中间内容
- **缓存效率**:每查询重 prefill 全文,KV cache 利用率低
- **隐私 / 更新**:每次都得把全文发给模型;文档变化需重发

### 30.9.4 RAG 的不替代价值

- **成本**:只送相关片段,prompt 短
- **新鲜**:换库不动模型
- **可追溯**:每条回答附引用
- **多文档**:无上限,只受向量库容量限制
- **私有**:可全本地

### 30.9.5 共识:互补

| 场景 | 推荐 |
| --- | --- |
| 单文档深问答(< 100 页) | Long-Context |
| 多文档 / 大库(GB+) | RAG |
| 实时变化数据 | RAG |
| 跨文档推理 | RAG + 长上下文混合 |
| Agent 自主调用 | RAG 作 tool |

最佳实践:**RAG 召回好片段,LLM 长上下文做高质量 reasoning**。两者协同 = 1+1>2。

---

## 30.10 评测:命中率 / 忠实度 / 端到端

### 30.10.1 检索评测

- **Recall@k**:正确 doc 是否出现在 top-k
- **MRR(Mean Reciprocal Rank)**:正确 doc 的倒数排名平均
- **nDCG@k**:考虑相关度梯度

### 30.10.2 生成评测

- **Faithfulness(忠实度)**:回答是否被检索文档支持(无幻觉)
- **Answer Relevance**:与问题相关性
- **Context Precision/Recall**:上下文用得对不对

工业工具:**RAGAS**、**TruLens**、**DeepEval**、**LlamaIndex Evaluator**。

### 30.10.3 端到端评测

最稳的是**人工评测**:对一批 query 标"完美答案",对比生成。代价高但唯一可靠。

---

## 30.11 完整代码:最小 RAG

```python
"""
最小可运行 RAG: 用 sentence-transformers + FAISS + 任何 OpenAI 兼容 LLM
pip install sentence-transformers faiss-cpu openai
"""

import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

EMB = "BAAI/bge-small-zh-v1.5"
emb_model = SentenceTransformer(EMB)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

# 1. 离线:索引文档
docs = [
    "司马懿在曹爽专权时期一直韬光养晦……",
    "高平陵之变发生在 249 年正月,司马懿趁曹爽出城祭陵……",
    "钢琴的标准音 A4 为 440 Hz……",
    "BERT 由 Google 在 2018 年提出,主要用于自然语言理解任务……",
]

def chunk(text, size=80, overlap=20):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)] or [text]

chunks = [c for d in docs for c in chunk(d)]
emb = emb_model.encode(chunks, normalize_embeddings=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1])      # cosine via IP(向量已归一化)
index.add(emb)

# 2. 在线:检索 + 生成
def rag(query, top_k=3):
    q_emb = emb_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q_emb, top_k)
    ctx = "\n\n".join(f"[doc {i}] {chunks[i]}" for i in idx[0])
    prompt = f"""请仅根据以下资料回答问题。不知道就回答"资料中未提及"。

资料:
{ctx}

问题: {query}
"""
    resp = client.chat.completions.create(
        model="local",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content

print(rag("高平陵之变发生在什么时候?"))
```

这是最小 RAG。生产版需加:

- 文档清洗、表格 / 图像抽取
- 结构感知 chunking + parent-child
- bge-reranker-v2-m3 二段精排
- 多向量库切片 + ANN 索引(HNSW)
- Streaming + 引用渲染
- 评测 + AB 测试

---

## 🎨 30.11.A 通俗类比合集

### 类比一: 大学考试 + 开卷 📖

> LLM 是聪明的考生,RAG 是"开卷 + 划重点"。

- **训练知识** = 考生的脑内记忆,考试前冻结
- **Chunking** = 把教材切成知识点
- **Embedding 入库** = 给每个知识点贴关键词标签
- **检索** = 考题来了,翻书找最相关的几页
- **Rerank** = 同学翻完后,你再人工挑出最相关的 2 页
- **HyDE** = 你先**自己假设答案**,再去翻"和这个答案最像"的页
- **Long-Context** = 整本书全摊在桌上,但找半天找不到关键页(lost-in-the-middle)

### 类比二: 公司客服中心 ☎️

> LLM 是接线员,RAG 是知识库系统。

- **知识库** = 公司所有产品文档,持续更新
- **Embedding** = 给文档自动建索引
- **检索** = 客户来电时,系统瞬间推 5 篇相关文档
- **Rerank** = 再用更精细的算法挑出 1 篇最贴切
- **LLM** = 接线员据此回答客户
- **Long-Context only** = 让接线员每次接电话前**重读整本 1000 页手册**——慢且贵
- **Citation** = 接线员附带"详见手册第 X 页",顾客可追查

### 类比三: 律师查案例 ⚖️

> 律师(LLM)+ 法律检索系统(RAG)。

- **判例库** = 千万级案件文本
- **Chunking** = 把每个案件切成"判决要旨 + 事实 + 法条"
- **稀疏 BM25** = 按"案由 / 法条号"关键词检索
- **稠密 embedding** = 按"案情语义相似"检索
- **Hybrid** = 律师两条线都用,合并取最相关
- **Rerank** = 助理把候选案例细读,标"高度相关"前 3 篇
- **LLM 生成** = 律师据此撰写法律意见
- **HyDE** = 先脑补"如果有这种判例会怎么判",再据此找

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 📖 开卷考 | **检索 + 答题** | RAG 完整管线 |
| ☎️ 客服 | **知识库可更新** | RAG vs Long-Context |
| ⚖️ 律师 | **多检索策略** | Hybrid + Rerank |

**记忆口诀**:
> **"切块入库,检索 + 重排,LLM 据资料答,Citation 给来源;Long-Context 是开整本书,RAG 是开关键页。"**

---

## 30.12 常见疑问

### Q1: chunk size 究竟选多大?
**A**: 没有银弹,先用 500-800 字符 + 100 重叠做 baseline。结构化文档(API doc / 法律 / 学术)可按章节切,非结构化用语义切。

### Q2: bge-m3 vs OpenAI embedding 哪个好?
**A**: bge-m3 在中文与多语言上**与 OpenAI 持平或更强**,且开源、免费、可私有部署。英文 + 通用查询 OpenAI 略稳。

### Q3: Hybrid 必须要吗?
**A**: 强烈推荐。BM25 几乎无成本(传统 IR 库),搭配 dense 检索召回普遍提升 10%+。

### Q4: Reranker 性价比?
**A**: 高。开源 reranker(bge-reranker)推理几毫秒一对,通常对 Top-50 → Top-5 跑;最终回答质量提升明显。延迟可接受。

### Q5: RAG 是不是死了(因为长上下文)?
**A**: 完全没有。生产 RAG 在 2024-2026 反而更繁荣——成本与可控性优势在企业场景不可替代。长上下文是 RAG 的好搭档,不是替代品。

---

## 30.13 本章小结

### 核心流水线

```text
Doc → Chunk → Embedding → Vector DB
Query → Embedding + (BM25) → Hybrid Retrieve Top-k → Rerank → Prompt → LLM
```

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **Chunking** | 500-800 字符 + 重叠,父子结构最稳 |
| **Embedding** | bge-m3 / OpenAI-3,中文优先 bge |
| **Hybrid** | 稀疏 + 稠密,RRF 融合 |
| **Reranker** | cross-encoder 精排,显著提升命中 |
| **HyDE / Multi-Query** | 高级技巧 |
| **vs Long-Context** | 互补,不替代 |
| **评测** | Recall@k + 忠实度 + 端到端 |

### 一句话总结

> **RAG 是"先检索后生成":离线把文档切块入向量库,在线对 query 做 embedding + BM25 hybrid 检索 + reranker 精排,拼成 prompt 喂 LLM;它让 LLM 拥有可更新、可追溯、可私有的外部知识,是企业 LLM 落地的事实标准——而长上下文 LLM 是它的好搭档,而非取代者。**

---

## 30.14 延伸阅读

### 必读论文

1. Lewis et al., 2020. *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*. NeurIPS. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
2. Robertson & Zaragoza, 2009. *"The Probabilistic Relevance Framework: BM25 and Beyond"*
3. Karpukhin et al., 2020. *"Dense Passage Retrieval"*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
4. Gao et al., 2022. *"HyDE: Precise Zero-Shot Dense Retrieval"*. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

### 进阶资源

5. *"BGE"* 系列 BAAI 模型卡 + 论文
6. LangChain / LlamaIndex 文档
7. RAGAS / TruLens 评测库
8. Microsoft GraphRAG
9. *"Lost in the Middle"* (Liu et al., 2023) - 长上下文位置 bias

---

## 30.15 实战练习

### 练习 1 (★)

跑通 §30.11 最小 RAG demo,用一份 PDF 做问答。

### 练习 2 (★★)

实现 Hybrid:用 `rank_bm25` 算 BM25 分,与 dense cosine 用 RRF 融合;对比单一检索 recall@10。

### 练习 3 (★★)

接入 bge-reranker-v2-m3:向量检索 Top-50 → reranker → Top-5,对比有 / 无 reranker 的最终回答正确率。

### 练习 4 (★★★)

实现 HyDE:让 LLM 先伪造一段答案,用其 embedding 检索;在你的语料上对比与朴素 RAG 的差异。

### 练习 5 (★★★)

设计长上下文 vs RAG 对照实验:同一本 100 页电子书,分别用 (a) RAG-Top5、(b) 长上下文整本,跑 50 个问答题,对比正确率 / 延迟 / 成本。

---

## 章节交叉引用

- 前置: [§01 向量内积](../Part_I_数学与基础/01_线性代数与矩阵运算.md), [§05 Tokenization](../Part_II_Transformer架构/05_Tokenization.md)
- 后续: [§31 Agent](31_Agent.md)(检索是核心 tool), [§32 Prompt 工程](32_Prompt工程.md)
- 相关: [§19 KV Cache](../Part_IV_推理与部署/19_KV_Cache与显存管理.md)(Prefix Cache 与 RAG), [§29 多模态](29_多模态模型.md)(多模态 RAG)

---

*下一章: §31 Agent*
