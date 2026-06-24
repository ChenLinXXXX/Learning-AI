# 05 Tokenization(分词:BPE / BBPE / SentencePiece / Tokenizer 实战)

> **章节定位**: Part II 起点,模型与文本之间的"翻译层"
> **预计阅读时间**: 50-70 分钟
> **难度**: ★★☆☆☆
> **前置知识**: 基础概率(频率统计)、UTF-8 编码常识
> **后续依赖**: §06(Embedding)、§11(数据准备)、§22(本地部署)、§32(Prompt 工程)

---

## 目录

- [摘要](#摘要)
- [学习目标](#学习目标)
- [5.1 为什么需要 Tokenization?](#51-为什么需要-tokenization)
- [5.2 三种基本粒度:Word / Char / Subword](#52-三种基本粒度word--char--subword)
- [5.3 BPE:Byte-Pair Encoding 算法](#53-bpebyte-pair-encoding-算法)
- [5.4 BBPE:Byte-level BPE(GPT-2 起的事实标准)](#54-bbpebyte-level-bpegpt-2-起的事实标准)
- [5.5 WordPiece 与 Unigram(SentencePiece)](#55-wordpiece-与-unigramsentencepiece)
- [5.6 完整代码实现:从零写一个 BPE](#56-完整代码实现从零写一个-bpe)
- [5.7 复杂度与工程细节](#57-复杂度与工程细节)
- [5.8 特殊 token、模板与对话格式](#58-特殊-token模板与对话格式)
- [5.9 跨语言、跨模型的 token 经济学](#59-跨语言跨模型的-token-经济学)
- [🎨 5.9.A 通俗类比合集](#-59a-通俗类比合集)
- [5.10 常见疑问](#510-常见疑问)
- [5.11 本章小结](#511-本章小结)
- [5.12 延伸阅读](#512-延伸阅读)
- [5.13 实战练习](#513-实战练习)
- [章节交叉引用](#章节交叉引用)

---

## 摘要

Tokenization(分词)是 LLM 与文本世界之间的**唯一接口**:文本进入模型前必须先被切成 token 序列再映射成整数 ID,模型输出的整数再被反向解码回文本。本章系统介绍:

- 为什么 LLM 普遍采用 **subword**(子词)而非"字"或"词"
- **BPE / BBPE** 的训练算法与推理算法,以及为什么它们能在覆盖 + 压缩之间取得平衡
- **WordPiece**(BERT)与 **Unigram**(SentencePiece、LLaMA、DeepSeek 默认)之间的根本差异
- 从零实现一个 30 行的 BPE,理解"merge rules"的本质
- 工程上必懂的细节:特殊 token、ChatTemplate、跨语言"token 通胀"问题

Tokenization 看似只是预处理,但它直接影响**上下文长度、推理成本、模型多语言能力、prompt 设计**。

---

## 学习目标

完成本章后,你能够:

- ✅ 解释为什么 LLM 都用 subword 而非 word/char (§5.2)
- ✅ 写出 BPE 训练过程的伪代码 (§5.3.2)
- ✅ 说出 BBPE vs BPE 的关键差别——以及为什么 BBPE 完全无 OOV (§5.4)
- ✅ 区分 WordPiece、Unigram 的目标函数 (§5.5)
- ✅ 用 ≤ 50 行 Python 实现一个能 fit + encode 的 BPE (§5.6)
- ✅ 估算同一段中文在 GPT-4o / LLaMA / DeepSeek tokenizer 下的 token 数差异 (§5.9)

---

## 5.1 为什么需要 Tokenization?

### 5.1.1 模型只懂数字

神经网络的输入必须是数值张量。给定字符串 `"我喜欢苹果"`,模型不能直接吃,需要某种映射:

$$
\text{tokenize}: \text{string} \to \mathbb{Z}^n \tag{5.1}
$$

每个整数 ID 之后会通过 embedding 矩阵(§06)映射成稠密向量。

### 5.1.2 切分粒度决定一切

同样一句话,切分粒度直接决定:

- **序列长度 $n$**: 影响 Attention 的 $O(n^2)$ 与 KV cache 体积
- **词表大小 $V$**: 影响 embedding 与 LM head 的参数量(≈ $2 V d_{\text{model}}$)
- **OOV(未登录词)**: 是否能表达任意输入
- **语义粒度**: 一个 token 承载多少含义

Tokenization 是这四者之间的**工程取舍**。

---

## 5.2 三种基本粒度:Word / Char / Subword

| 粒度 | 例:`"unaffordable"` | 词表大小 | OOV | 序列长度 |
| --- | --- | --- | --- | --- |
| **Word** | `["unaffordable"]` | 极大(数百万) | 多 | 短 |
| **Char** | `["u","n","a",...,"e"]` | 极小(~256) | 无 | 长(每字符一 token) |
| **Subword** | `["un","afford","able"]` | 中(~5万-20万) | 极少 / 无 | 中 |

### 5.2.1 Word 级的问题

英文按空格切看似自然,但:

- 词表爆炸(英语主流词典 50 万+,加上各种变体上百万)
- OOV 灾难:训练时没见过的词只能映射成 `<UNK>`,丢失信息
- 中文 / 日文 / 韩文 没有显式空格,无法直接 word 化

### 5.2.2 Char 级的问题

- 序列变长 5-10 倍,推理算力暴涨
- 单字符语义信息稀薄,模型要从更长依赖中"自己组词",学习困难
- 仍然难以表达 emoji、不常见 Unicode 等

### 5.2.3 Subword 的核心思想

> **常见词作为整块,罕见词被拆成更短的"零件"。**

效果:**词表中等、序列适中、零 OOV、语义粒度有层次**。从 2016 年 BPE 应用于 NMT 起,subword 成为主流,所有现代 LLM 都基于它。

---

## 5.3 BPE:Byte-Pair Encoding 算法

### 5.3.1 起源

BPE 最早是 1994 年的**数据压缩算法**(Gage),核心想法:**反复找出最频繁的相邻字节对,合并成新符号,直到达到目标符号数**。

Sennrich et al.(2016)把它搬到 NMT 上,从此成为 NLP 的标配。

### 5.3.2 训练算法

给定大语料 $\mathcal{C}$ 和目标词表大小 $V$:

```text
1. 初始化:将每个 word 拆成字符序列,末尾加 </w>(或特殊边界)
   例: "low" -> ['l','o','w','</w>']

2. 统计当前所有相邻 token pair 的频率

3. 选出频率最高的 pair (a, b),合并:
   a. 把所有 (a, b) 替换成新 token "ab"
   b. 记录这一条 merge rule: (a, b) -> "ab"

4. 重复步骤 2-3,直到词表大小达到 V
```

### 5.3.3 推理算法(encode)

新字符串 $s$ 来了:

```text
1. 按字符切分
2. 按训练时记录的 merge rules 顺序,从前往后扫描,能合并就合并
3. 直到没有 rule 可应用,输出 token 序列
```

注意 **merge 顺序固定**,这保证训练与推理一致。

### 5.3.4 一个微型例子

语料 `low low lower lowest` 拆成字符:

| 步骤 | 高频 pair | 合并后 |
| --- | --- | --- |
| 0 | — | `l o w</w>, l o w</w>, l o w e r</w>, l o w e s t</w>` |
| 1 | `(l, o)` 出现 4 次 | `lo w</w>, lo w</w>, lo w e r</w>, lo w e s t</w>` |
| 2 | `(lo, w)` 出现 4 次 | `low</w>, low</w>, low e r</w>, low e s t</w>` |

`low` 成为整块,`lower / lowest` 仍能用 `low + e + r/s + t` 组装出来——**这就是"零 OOV"的来源**。

---

## 5.4 BBPE:Byte-level BPE(GPT-2 起的事实标准)

### 5.4.1 BPE 的隐患:Unicode 与 OOV

朴素 BPE 把字符当原子。但 Unicode 有 14 万+ 字符,如果训练语料没出现过某个汉字、emoji、阿拉伯字母,它**仍然会变成 `<UNK>`**。

### 5.4.2 BBPE 的思想

> **把所有字符串先转 UTF-8 字节流,然后在 256 个字节上跑 BPE。**

字节集是有限闭集(256),任何 Unicode 字符串都能被 256 字节表达,因此 **BBPE 绝对零 OOV**。

GPT-2 首次采用,GPT-3、GPT-4、LLaMA、Mistral、Qwen、DeepSeek 全部沿用。

### 5.4.3 代价:中文 token 通胀

UTF-8 中:

- 1 个 ASCII 字符 = 1 字节
- 1 个常见汉字 = 3 字节
- 1 个 emoji = 4 字节

如果训练语料以英文为主,**中文很容易被拆成 2-3 个 token / 字**——这就是早期 GPT-3 中文巨贵的根源。

后来各家通过在多语言语料上重新训练 tokenizer,把常见汉字 / 词组进词表,显著降低了中文 token 数。详见 §5.9。

---

## 5.5 WordPiece 与 Unigram(SentencePiece)

### 5.5.1 WordPiece(BERT)

与 BPE 几乎一致,但**合并准则不是频率而是似然增益**:

$$
\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \cdot \text{freq}(b)} \tag{5.2}
$$

意为:"合并后比单独出现的概率倍数",倾向于挑出"强搭配"。BERT 用它,但生成式 LLM 现在已很少用。

### 5.5.2 Unigram(SentencePiece)

Kudo, 2018。思路与 BPE 完全相反:

- BPE:**自底向上**,从字符开始一步步合并
- Unigram:**自顶向下**,先用大词表,逐步删除"对总似然贡献最小"的 token,直到达到目标大小

其目标函数是:

$$
\mathcal{L}(\theta) = \sum_{s \in \mathcal{C}} \log \sum_{\text{seg}} \prod_i p(t_i; \theta) \tag{5.3}
$$

其中 $\text{seg}$ 是所有可能的切分方式。Unigram 训练时考虑**多切分概率**,推理时用 Viterbi 选最大概率切分。

### 5.5.3 SentencePiece 工具

Google 的 SentencePiece 是一套**与语言无关、不依赖空格预切分**的实现,既支持 BPE 也支持 Unigram。

特点:

- 把空格当成普通符号(`▁`)处理 → **无需语言相关的预处理**
- 直接处理 raw text → 适合多语言混合
- LLaMA、Mistral、DeepSeek、Qwen 都用 SentencePiece(多为 BPE 模式)

### 5.5.4 选型对照

| 算法 | 训练方向 | 准则 | 代表模型 |
| --- | --- | --- | --- |
| BPE | 自底向上 | 频率 | GPT-2/3/4 |
| BBPE | 字节级 BPE | 频率 | GPT-2+, LLaMA, DeepSeek |
| WordPiece | 自底向上 | 似然增益 | BERT |
| Unigram | 自顶向下 | 整体语言模型似然 | T5, 部分 SentencePiece 模型 |

---

## 5.6 完整代码实现:从零写一个 BPE

下面是一个**最小可运行**的 BPE(约 50 行),便于理解算法本质:

```python
from collections import Counter, defaultdict
from typing import List, Tuple


class MiniBPE:
    def __init__(self, vocab_size: int = 200):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: set = set()

    def _get_stats(self, words):
        pairs = Counter()
        for word, freq in words.items():
            tokens = word.split()
            for a, b in zip(tokens[:-1], tokens[1:]):
                pairs[(a, b)] += freq
        return pairs

    def _merge_pair(self, words, pair):
        a, b = pair
        bigram = f"{a} {b}"
        new_token = a + b
        new_words = {}
        for word, freq in words.items():
            new_words[word.replace(bigram, new_token)] = freq
        return new_words, new_token

    def fit(self, corpus: List[str]):
        # 初始化:每个 word -> 字符序列(空格分隔)
        words = Counter()
        for line in corpus:
            for w in line.split():
                words[" ".join(list(w)) + " </w>"] += 1
        self.vocab = {ch for w in words for ch in w.split()}

        # 反复合并最频繁 pair
        while len(self.vocab) < self.vocab_size:
            stats = self._get_stats(words)
            if not stats:
                break
            best = stats.most_common(1)[0][0]
            words, new_token = self._merge_pair(words, best)
            self.merges.append(best)
            self.vocab.add(new_token)

    def encode(self, text: str) -> List[str]:
        tokens = []
        for w in text.split():
            piece = " ".join(list(w)) + " </w>"
            for a, b in self.merges:
                piece = piece.replace(f"{a} {b}", a + b)
            tokens.extend(piece.split())
        return tokens


# === 测试 ===
if __name__ == "__main__":
    corpus = ["low low low low low lower lowest newest newest widest"]
    bpe = MiniBPE(vocab_size=30)
    bpe.fit(corpus)
    print(bpe.encode("lowest newer"))
```

预期输出会把 `lowest` 切成 `["low","est","</w>"]` 等组合——你可以打印 `bpe.merges` 看模型学到了哪些 merge rules。

> 生产级实现请用 [tokenizers](https://github.com/huggingface/tokenizers)(Rust 写,百倍速度)或 SentencePiece。

---

## 5.7 复杂度与工程细节

### 5.7.1 训练复杂度

朴素 BPE 训练:每轮 $O(N)$ 扫描语料,做 $V$ 轮 → $O(NV)$。HuggingFace `tokenizers` 用 Rust + 增量 stats,大语料上可数小时训出百万级词表。

### 5.7.2 推理复杂度

预编译 merge rules 后,encode 一段长 $L$ 的文本约 $O(L \log V)$(按 rules trie / priority queue)。

### 5.7.3 工程细节

- **预 tokenization (pre-tokenization)**: 真实 tokenizer 会先按空格 / 标点做粗切分,再 BPE。GPT 用 regex,LLaMA 用 SentencePiece 自带规则。
- **NFC / NFKC 归一化**: 把全角半角、组合字符统一,避免视觉相同但字节不同的字符产生不同 token。
- **byte fallback**: 没在 vocab 里的 Unicode 字符回退到字节,确保零 OOV(SentencePiece BPE 的标配)。

---

## 5.8 特殊 token、模板与对话格式

### 5.8.1 常见特殊 token

| Token | 用途 |
| --- | --- |
| `<BOS>` / `<s>` | 序列起始 |
| `<EOS>` / `</s>` | 序列结束(也作为"停止信号") |
| `<PAD>` | batch 中对齐填充 |
| `<UNK>` | 未知 token(BBPE 中已废弃) |
| `<|im_start|>` / `<|im_end|>` | OpenAI Chat 风格分隔 |
| `<|user|>` / `<|assistant|>` | 多角色对话标记 |

### 5.8.2 ChatTemplate

现代对话模型把"角色 + 内容"用特殊 token 包起来,例:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好!<|im_end|>
```

不同模型的 chat template **不一样**(LLaMA-3、Mistral、Qwen、DeepSeek 各有标准),调用本地模型时必须严格按模型卡里给的模板拼,**否则模型表现会显著退化**。HuggingFace `tokenizer.apply_chat_template(...)` 会替你处理这些差异。

### 5.8.3 对 prompt 工程的影响

特殊 token **占 token 数**,长 system prompt 会显著吃掉上下文。对 RAG / Agent 这种 prompt 长的场景,优化模板格式也是省 token 的关键(详见 §32)。

---

## 5.9 跨语言、跨模型的 token 经济学

### 5.9.1 同一句话,不同模型的 token 数

以"今天天气真好,我们去公园散步吧。"为例(实测近似):

| Tokenizer | token 数 |
| --- | --- |
| GPT-3.5 (cl100k_base) | ≈ 22 |
| GPT-4o (o200k_base) | ≈ 11 |
| LLaMA-2 | ≈ 25 |
| LLaMA-3 | ≈ 13 |
| DeepSeek-V3 | ≈ 11 |
| Qwen-2.5 | ≈ 10 |

**结论**:中文友好型 tokenizer 把同句压成接近一半,意味着同样的上下文窗口能装更多内容、推理也更便宜。

### 5.9.2 影响

- **API 成本**:OpenAI 按 token 收费,Tokenizer 决定一段文本 50% 的成本差
- **上下文有效长度**:128k 窗口对中文用户,实际能用的"字数"还要乘以一个折扣
- **多语言公平**:早期 BBPE 的"英文便宜中文贵"被批评为隐形不公,现在已大幅改进

### 5.9.3 怎么估算成本?

经验粗算:

| 语言 | 字符 / token(现代 LLM) |
| --- | --- |
| 英文 | ≈ 4 字符 / token |
| 中文 | ≈ 1.5 字符 / token |
| 代码 | ≈ 3-4 字符 / token |

更准用 `tiktoken` / `tokenizers` 实测。

---

## 🎨 5.9.A 通俗类比合集

抽象的子词切分,其实生活里到处都是。

### 类比一: 乐高积木 🧱

> 拼一座城堡有几种方案:整块预制、零散颗粒、混合预制件。

- **Word**:整块"城堡"预制 → 库存巨大,没库存的造型直接 GG(OOV)
- **Char**:全部用最小颗粒 → 灵活但拼一座城堡要几千块
- **Subword(BPE)**:常用的"墙体"、"塔尖"做成中等预制件,稀有造型再用颗粒拼 → **库存适中、灵活、可表达任意造型**

**BBPE = 砖块退化到"字节级最小颗粒"**:不管你想拼什么稀奇造型(emoji、藏文、楔形文字),都能用 256 种基础颗粒搭出来,零 OOV。

### 类比二: 中药铺的"分装" 🌿

> 同一味药材,可以整株卖、可以片剂分装、可以打粉。

- **整株卖** = Word:好认,但每株都是独立 SKU
- **打粉散装** = Char:任何方子都能配,但配一副要称几十次粉末
- **片剂分装** = Subword:常用的(当归、黄芪)按片卖,稀有的按小袋分装 → 兼顾效率与灵活

**Tokenization 训练 = 中药铺老板根据这一年开出的方子频率,决定哪些做成片剂、哪些只散卖**。

### 类比三: 城市地铁系统 🚇

> 把一段话当作一次"通勤路线"。

- **Word**:每一站都是一个完整地名(北京西站、国贸…),站点数少但站名长
- **Char**:每个站只能停一个字,要无数次换乘
- **Subword**:站点是常见的词块(`国贸`、`大兴`、`机场`),罕见地名拆开换乘 → **总通勤时长最短**

**BBPE 的字节 fallback** 就像"如果你要去一个偏远的、没有命名站点的小村子,可以买一张'字节车票',无缝转车到达"。

### 三个类比对比

| 类比 | 强调什么 | 适合记忆什么 |
| --- | --- | --- |
| 🧱 乐高积木 | **粒度 vs 表达力** | Subword 的折中本质 |
| 🌿 中药分装 | **按频率定颗粒** | BPE 训练过程的统计驱动 |
| 🚇 地铁系统 | **效率与覆盖兼顾** | 跨语言 token 数差异 |

**记忆口诀**:
> **"高频整块卖,低频拆开拼,字节兜底零遗漏。"**

---

## 5.10 常见疑问

### Q1: 中文一定要专门训练 tokenizer 吗?
**A**: 看场景。如果用通用模型(GPT-4o、DeepSeek-V3),它们已经在中文上充分训练,token 效率不错;但如果自己做行业模型(金融、医疗术语),自训 tokenizer 把术语整块入词表能显著降本。

### Q2: token 数和"字数"什么关系?
**A**: 没有严格对应。粗估见 §5.9.3。准确做法:直接 `len(tokenizer.encode(text))`。

### Q3: 我换一个 tokenizer,模型还能用吗?
**A**: 不行。Tokenizer 与 embedding / LM head 是**绑定**的——更换会让所有 ID 错位。要换 tokenizer 必须同时重训(或至少 fine-tune)embedding 与 LM head。

### Q4: 为什么 BBPE 的词表里我看到一堆"乱码"?
**A**: 因为它在字节级训练,Unicode 字符的中间字节单独看是"半个字符",`tokenizers` 会把不可打印字节用特殊字符串(如 `Ġ` 表示空格)替换显示——这不是 bug。

### Q5: vocab_size 越大越好吗?
**A**: 不是。大词表 → embedding/LM head 参数膨胀($\approx 2 V d$),也分散了每个 token 的训练样本。主流取舍:**32k(LLaMA-2)→ 128k(LLaMA-3)→ 200k(GPT-4o)→ 129k(DeepSeek-V3)**。多语言模型一般更大。

---

## 5.11 本章小结

### 核心算法

**BPE 训练循环**:

$$
\text{repeat: } \arg\max_{(a,b)} \text{freq}(a,b) \to \text{merge to } ab \text{ until } |V| = V_{\text{target}}
$$

**Unigram 训练目标**:

$$
\max_{\theta} \sum_{s \in \mathcal{C}} \log \sum_{\text{seg}} \prod_i p(t_i; \theta)
$$

### 核心要点

| 要点 | 关键 |
| --- | --- |
| **粒度选择** | Subword 是当今 LLM 唯一主流 |
| **算法** | BPE 频率合并 / Unigram 似然剪枝 |
| **OOV** | BBPE 通过字节级彻底消除 |
| **特殊 token** | 必须按模型卡的 chat template 用 |
| **跨语言** | 中文/小语种 token 通胀是真实成本 |

### 一句话总结

> **Tokenization 是文本与模型之间唯一的"翻译层":通过自底向上(BPE)或自顶向下(Unigram)的统计算法,把任意字符串切成有限词表内的 subword 序列,再字节级兜底以保证零 OOV。它决定了上下文长度、推理成本与多语言公平。**

---

## 5.12 延伸阅读

### 必读论文

1. Sennrich et al., 2016. *"Neural Machine Translation of Rare Words with Subword Units"*. ACL. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)(BPE 进 NMT)
2. Radford et al., 2019. *"Language Models are Unsupervised Multitask Learners"* (GPT-2 技术报告,BBPE 首次工程化)
3. Kudo, 2018. *"Subword Regularization"*. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959)(Unigram 算法)

### 进阶论文

4. Kudo & Richardson, 2018. *"SentencePiece: A simple and language independent subword tokenizer"*. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
5. Wu et al., 2016. *"Google's Neural Machine Translation System"*(WordPiece)
6. Petrov et al., 2023. *"Language Model Tokenizers Introduce Unfairness Between Languages"*. [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)

### 推荐资源

- HuggingFace Course: *"The Tokenizer's Role"*
- OpenAI Cookbook: *"How to count tokens with tiktoken"*
- Karpathy: *"Let's build the GPT Tokenizer"* YouTube

---

## 5.13 实战练习

### 练习 1 (★)

用 `tiktoken` 与 `transformers.AutoTokenizer` 分别加载 `gpt-4o`、`deepseek-ai/DeepSeek-V3`、`meta-llama/Meta-Llama-3-8B` 的 tokenizer,对同一段中英混合文本统计 token 数,验证 §5.9.1 的差异。

### 练习 2 (★★)

把 §5.6 的 `MiniBPE` 改成 **byte-level**(BBPE),并验证它能编码任意 emoji。

### 练习 3 (★★)

实现一个 Unigram 训练器(用 EM 算法迭代估计 $p(t)$),并在 §5.6 的同一语料上对比 BPE 与 Unigram 学到的切分差异。

### 练习 4 (★★★)

写一个工具:输入一个 chat 消息列表,用 HuggingFace `apply_chat_template` 输出 3 个不同模型(LLaMA-3 / Qwen / DeepSeek)的拼接结果,直观比较模板差异与 token 占比。

### 练习 5 (★★★)

设计实验:固定模型 LLaMA-2-7B,只替换 tokenizer 为"中文增强版"(扩词表),观察推理速度 / 输出质量 / 显存的变化。讨论换 tokenizer 必须做哪些工程改造。

---

## 章节交叉引用

- 前置: 无(Part II 起点)
- 后续: [§06 Embedding 与位置编码](06_Embedding与位置编码.md), [§11 数据准备](../Part_III_训练原理/11_数据准备与清洗.md), [§32 Prompt 工程](../Part_VI_应用前沿/32_Prompt工程.md)
- 相关: [§07 Self-Attention](07_Self_Attention机制.md)(token 数决定复杂度), [§19 KV Cache](../Part_IV_推理与部署/19_KV_Cache与显存管理.md)

---

*下一章: §06 Embedding 与位置编码(RoPE / YaRN / ALiBi)*
