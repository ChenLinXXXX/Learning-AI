# 附录 B — 常用公式速查

> **范围**: 整合本书核心公式,按主题分组,**只列公式 + 一句话用途 + 章节回链**。
> **用法**: 重看代码 / 论文时,公式回忆速查。

---

## 目录

- [B.1 线性代数与张量](#b1-线性代数与张量)
- [B.2 概率与信息论](#b2-概率与信息论)
- [B.3 反向传播](#b3-反向传播)
- [B.4 优化算法](#b4-优化算法)
- [B.5 Attention 家族](#b5-attention-家族)
- [B.6 Embedding / 位置编码](#b6-embedding--位置编码)
- [B.7 FFN / MoE / 归一化](#b7-ffn--moe--归一化)
- [B.8 训练目标](#b8-训练目标)
- [B.9 偏好对齐](#b9-偏好对齐)
- [B.10 Scaling Law](#b10-scaling-law)
- [B.11 推理 / KV / 采样](#b11-推理--kv--采样)
- [B.12 RAG / Retrieval](#b12-rag--retrieval)

---

## B.1 线性代数与张量

**矩阵乘**:

$$
(AB)_{ik} = \sum_j a_{ij} b_{jk}, \quad \text{FLOPs} = 2 m n p
$$

— §01.2

**余弦相似度**:

$$
\cos\theta = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$

— §01.4.2

**SVD**:

$$
A = U \Sigma V^\top, \quad A_r = U_{[:,:r]} \Sigma_{[:r,:r]} V_{[:,:r]}^\top \;\text{(最优秩-}r\text{ 近似)}
$$

— §01.5

---

## B.2 概率与信息论

**熵**:

$$
H(p) = -\sum_x p(x) \log p(x)
$$

**交叉熵**:

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

**KL 散度**:

$$
\text{KL}(p \| q) = \sum_x p(x) \log\frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

**Softmax(带温度)**:

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

**Perplexity**:

$$
\text{PPL} = \exp(H) = \exp(\text{CE})
$$

— §02.5-6

---

## B.3 反向传播

**链式法则(标量)**:

$$
\frac{dy}{dx} = \sum_i \frac{\partial y}{\partial u_i} \cdot \frac{du_i}{dx}
$$

**矩阵乘 vjp**($Y = X W$):

$$
\frac{\partial \mathcal{L}}{\partial W} = X^\top \frac{\partial \mathcal{L}}{\partial Y}, \quad \frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} W^\top
$$

**Softmax + CE 反向**:

$$
\frac{\partial \mathcal{L}}{\partial z} = p - y
$$

**梯度裁剪**:

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|_2}\right)
$$

— §03

---

## B.4 优化算法

**AdamW**:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t), \quad \hat v_t = v_t / (1 - \beta_2^t) \\
\theta &\leftarrow \theta - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta \right)
\end{aligned}
$$

**Cosine 学习率**:

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\!\left(1 + \cos\pi\frac{t - T_w}{T - T_w}\right)
$$

— §04

---

## B.5 Attention 家族

**Scaled Dot-Product**:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

**Multi-Head**:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

**MLA(KV 压缩 + decoupled RoPE)**:

$$
c^{KV}_t = W^{DKV} h_t, \quad k_{t,i} = \begin{bmatrix} W^{UK}_i c^{KV}_t \\ \text{RoPE}(W^{KR} h_t) \end{bmatrix}
$$

— §07 / §24

---

## B.6 Embedding / 位置编码

**Sinusoidal**:

$$
P_{i, 2k} = \sin(i / 10000^{2k/d}), \quad P_{i, 2k+1} = \cos(i / 10000^{2k/d})
$$

**RoPE 内积只依赖相对距离**:

$$
\langle R_i q, R_j k \rangle = q^\top R_{j-i} k
$$

**ALiBi**:

$$
A_{ij} \mathrel{+}= -m_h \cdot |i - j|
$$

— §06

---

## B.7 FFN / MoE / 归一化

**Standard FFN**: $\text{FFN}(x) = W_2 \sigma(W_1 x)$

**SwiGLU FFN**:

$$
\text{FFN}_{\text{SwiGLU}}(x) = W_\text{down}\big[\text{Swish}(W_\text{gate} x) \odot (W_\text{up} x)\big]
$$

**SwiGLU 与 ReLU 参数持平**: $d_{\text{ff}} \approx \tfrac{8}{3} d$

**DeepSeekMoE**:

$$
\text{MoE}(x) = \sum_s E^{\text{shared}}_s(x) + \sum_{i \in \mathcal{T}_k(x)} g_i(x)\,E_i(x)
$$

**Aux-Loss-Free Top-k**:

$$
\mathcal{T}_k = \text{TopK}_i(z_i + b_i), \quad g_i = \text{softmax}(z_i)_{i \in \mathcal{T}_k}
$$

**LayerNorm / RMSNorm**:

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \quad \text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum x_i^2 + \epsilon}}
$$

— §08 / §09 / §25

---

## B.8 训练目标

**NTP**:

$$
\mathcal{L}_{\text{NTP}} = -\frac{1}{T-1} \sum_t \log p_\theta(x_{t+1} \mid x_{\le t})
$$

**MLM**:

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log p_\theta(x_t \mid x_{\setminus \mathcal{M}})
$$

**MTP**:

$$
\mathcal{L}_{\text{MTP}} = \mathcal{L}_{\text{NTP}} + \lambda \cdot \tfrac{1}{N-1}\sum_{k=2}^N \text{CE}_k
$$

— §12

---

## B.9 偏好对齐

**RM 训练(Bradley-Terry)**:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\phi(x, y_c) - r_\phi(x, y_r)\big)
$$

**PPO + KL 锚**:

$$
\max_\theta \; \mathbb{E}_{x, y\sim\pi_\theta}\big[r(x, y)\big] - \beta\,\text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

**DPO**:

$$
\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta \log\frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log\frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)
$$

**GRPO**(组内归一 advantage):

$$
A_i = \frac{r_i - \text{mean}\{r_j\}}{\text{std}\{r_j\} + \epsilon}
$$

$$
\mathcal{L}_{\text{GRPO}} = -\tfrac{1}{G}\sum_i \min\!\big(\rho_i A_i,\, \text{clip}(\rho_i, 1\!-\!\epsilon, 1\!+\!\epsilon)\, A_i\big) + \beta\,\text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

— §16 / §28

---

## B.10 Scaling Law

**Chinchilla**:

$$
L(N, D) \approx L_\infty + A N^{-\alpha} + B D^{-\beta}, \quad D^* \approx 20 N^*
$$

**训练算力**:

$$
C \approx 6 P D
$$

**Inference-aware 总成本**:

$$
C_\text{total} = 6 P D + 2 P D_\text{infer}
$$

**MoE 有效参数(经验)**:

$$
P_\text{eff} \approx \sqrt{P_\text{total} \cdot P_\text{act}}
$$

— §17

---

## B.11 推理 / KV / 采样

**KV cache 字节(单序列)**:

$$
\text{KV}_{\text{single}} = 2 \cdot L \cdot n \cdot h_{\text{kv}} \cdot d_k \cdot \text{bytes}
$$

**MLA KV(每 token 每层)**:

$$
\text{KV}_{\text{MLA}} = (d_c + d_h^{(R)}) \cdot \text{bytes}
$$

**Speculative 加速比(粗算)**:

$$
\text{Speedup} \approx \frac{1 + \alpha + \alpha^2 + \cdots + \alpha^K}{1 + K \cdot T_d / T_t}
$$

**Roofline**:

$$
\text{Time} \geq \max\!\left(\frac{\text{FLOPs}}{\text{TFLOPS}}, \frac{\text{Bytes}}{\text{HBM\_BW}}\right)
$$

**显存粗算(推理)**:

$$
\text{Mem} \approx \frac{P \cdot \text{bits/param}}{8} + \text{KV} + \text{overhead}
$$

— §18-22

---

## B.12 RAG / Retrieval

**BM25**(简化):

$$
\text{BM25}(q,d) = \sum_{t\in q} \text{IDF}(t) \cdot \frac{f(t,d)(k_1+1)}{f(t,d) + k_1(1 - b + b \cdot |d|/\bar L)}
$$

**RRF(rank fusion)**:

$$
\text{RRF}(d) = \sum_i \frac{1}{k + r_i(d)}
$$

**Hybrid 分数**:

$$
s(q, d) = \alpha \cdot \text{BM25}(q,d) + (1-\alpha) \cdot \cos(v_q, v_d)
$$

— §30
