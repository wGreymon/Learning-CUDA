# Flash Attention 中的 Tile 大小和重标定机制详解

## 一、基本概念

### 1.1 `block_size_q` 和 `block_size_kv` 的含义

**`block_size_q`**：Query 序列的分块大小（tile size）
- 表示每次处理多少个 query 位置
- 例如：`block_size_q = 32` 表示每次处理 32 个 query token

**`block_size_kv`**：Key/Value 序列的分块大小（tile size）
- 表示每次处理多少个 key/value 位置
- 例如：`block_size_kv = 64` 表示每次处理 64 个 key/value token

### 1.2 分块（Tiling）策略

Flash Attention 使用**分块计算**来避免存储完整的注意力矩阵：

```
完整注意力矩阵: QK^T = [L_q × L_k]  ← 内存占用 O(L_q × L_k)，太大！

分块计算:
  - 将 Q 分成多个块，每块 block_size_q 个 query
  - 将 K/V 分成多个块，每块 block_size_kv 个 key/value
  - 每次只计算一个小块: [block_size_q × block_size_kv]
  - 内存占用: O(block_size_q × block_size_kv)，可控！
```

**代码中的体现**：
```cpp
// 第 140-165 行：加载 Q tile
const int q_start = q_block_idx * block_size_q;
const int q_end = min(q_start + block_size_q, tgt_seq_len);
// 加载 block_size_q 个 query 到 shared memory

// 第 167-193 行：循环处理多个 KV tile
const int num_kv_blocks = (src_seq_len + block_size_kv - 1) / block_size_kv;
for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
    const int kv_start = kv_block_idx * block_size_kv;
    const int kv_end = min(kv_start + block_size_kv, src_seq_len);
    // 加载 block_size_kv 个 key/value 到 shared memory
    // 计算注意力分数: [block_size_q × block_size_kv]
}
```

## 二、重标定（Rescaling）机制

### 2.1 什么是重标定？

**重标定**是 Online Softmax 算法的核心机制，用于在分块计算过程中维护正确的 softmax 统计量。

### 2.2 为什么需要重标定？

在标准 softmax 中，我们需要：
1. 找到所有元素的最大值 `m = max(scores)`
2. 计算归一化因子 `l = sum(exp(scores - m))`
3. 计算输出 `output = sum(exp(scores - m) / l * values)`

但在分块计算中，我们**不能一次性看到所有元素**，需要：
- 逐步处理每个 KV tile
- 每次处理时，最大值可能会更新
- 需要重新调整（重标定）之前累积的结果

### 2.3 重标定过程详解

**步骤 1：计算当前 tile 的最大值**
```cpp
// 第 235-242 行
double m_ij = -INFINITY;
for (int kv_idx = 0; kv_idx < kv_block_len; kv_idx++) {
    double s_val = s_shared[q_idx * block_size_kv + kv_idx];
    m_ij = fmax(m_ij, s_val);  // 当前 tile 的最大值
}
```

**步骤 2：更新全局最大值**
```cpp
// 第 244 行
double m_old = m_shared[q_idx];  // 之前累积的最大值
double m_new = fmax(m_old, m_ij); // 新的全局最大值
```

**步骤 3：计算重标定因子**
```cpp
// 第 247 行
double alpha = exp(m_old - m_new);
```

**重标定因子的含义**：
- 如果 `m_new > m_old`（发现了更大的值）
- 那么 `alpha = exp(m_old - m_new) < 1`
- 需要**缩小**之前累积的值，因为之前用的是较小的 `m_old` 作为基准

**步骤 4：重标定历史累积值**
```cpp
// 第 267 行：重标定输出累积值
double o_new_val = alpha * o_old_val;  // 缩小之前的输出

// 第 289 行：重标定归一化因子
double l_new = alpha * l_old + p_sum;  // 缩小之前的归一化因子，加上新的
```

### 2.4 重标定示例

假设我们有两个 KV tile：

**Tile 1**：
- 最大值：`m_1 = 5.0`
- 累积输出：`o_1 = 10.0`
- 归一化因子：`l_1 = 2.0`

**Tile 2**：
- 最大值：`m_2 = 7.0`  ← 比 m_1 大！
- 需要重标定：
  - `alpha = exp(5.0 - 7.0) = exp(-2.0) ≈ 0.135`
  - `o_new = alpha * o_1 + new_contributions = 0.135 * 10.0 + ...`
  - `l_new = alpha * l_1 + new_sum = 0.135 * 2.0 + ...`

**为什么需要重标定？**
- 之前用 `m_1 = 5.0` 作为基准计算了 `exp(scores - 5.0)`
- 现在发现真正的最大值是 `m_2 = 7.0`
- 需要将所有之前的 `exp(scores - 5.0)` 转换为 `exp(scores - 7.0)`
- 转换公式：`exp(scores - 7.0) = exp(scores - 5.0) * exp(5.0 - 7.0) = exp(scores - 5.0) * alpha`

## 三、Tile 大小设计的意义

### 3.1 当前设计策略

```cpp
if constexpr (std::is_same_v<T, float>) {
    // float 路径：更大的 tile
    block_size_q  = (head_dim >= 128) ? 16 : ((head_dim > 64) ? 32 : 32);
    block_size_kv = (head_dim >= 128) ? 32 : ((head_dim > 64) ? 64 : 64);
} else {
    // half 路径：标准 tile
    block_size_q  = (head_dim >= 128) ? 8 : ((head_dim > 64) ? 8 : 16);
    block_size_kv = (head_dim >= 128) ? 16 : ((head_dim > 64) ? 32 : 32);
}
```

### 3.2 设计考虑因素

#### 1. **减少重标定次数**

**重标定次数 = KV tile 的数量 = `ceil(src_seq_len / block_size_kv)`**

- **更大的 `block_size_kv`** → **更少的 KV tile** → **更少的重标定次数**
- 每次重标定都会引入数值误差
- **减少重标定次数 = 提高数值精度**

**示例**：
```
src_seq_len = 1024

方案 A: block_size_kv = 32
  → num_kv_blocks = 1024 / 32 = 32 次重标定

方案 B: block_size_kv = 64
  → num_kv_blocks = 1024 / 64 = 16 次重标定  ← 精度更好！
```

#### 2. **内存限制**

Shared memory 大小限制（通常 48KB）：
```
内存占用 = 
  Q tile: block_size_q * head_dim * sizeof(T)
  K tile: block_size_kv * head_dim * sizeof(T)
  V tile: block_size_kv * head_dim * sizeof(T)
  Scores: block_size_q * block_size_kv * sizeof(double)
  O accum: block_size_q * head_dim * sizeof(double)
  M/L: block_size_q * sizeof(double) * 2
```

**更大的 tile** → **更多的内存占用**
- 需要在精度和内存之间平衡
- 代码中有自适应调整（第 400-406 行）

#### 3. **数据类型差异**

**Float 路径使用更大的 tile**：
- Float 精度要求更高（容差 1e-5）
- 需要减少重标定次数来提高精度
- 可以使用更大的 tile（因为 float 占用内存更少）

**Half 路径使用标准 tile**：
- Half 精度要求相对宽松（容差 0.02-0.05）
- 可以使用较小的 tile
- 节省内存，支持更大的模型

### 3.3 根据 head_dim 调整

```cpp
block_size_q  = (head_dim >= 128) ? 16 : ((head_dim > 64) ? 32 : 32);
block_size_kv = (head_dim >= 128) ? 32 : ((head_dim > 64) ? 64 : 64);
```

**重要发现：head_dim 越大，tile 越小！**

这是**内存限制**导致的，不是精度考虑。让我们看看内存占用：

**内存占用计算**：
```
总内存 = Q/K/V + Scores + O accumulator + M/L

Q/K/V: (block_size_q + 2×block_size_kv) × head_dim × sizeof(T)
Scores: block_size_q × block_size_kv × sizeof(double)
O: block_size_q × head_dim × sizeof(double)
M/L: 2 × block_size_q × sizeof(double)
```

**实际内存占用（float 类型）**：

| head_dim | Tile 大小 | 内存占用 | 是否超过 48KB？ |
|----------|----------|---------|----------------|
| 64 | 32×64 | **72.5 KB** | ❌ 超过 |
| 64 | 16×32 | **32.25 KB** | ✅ 安全 |
| 128 | 32×64 | **128.5 KB** | ❌ 超过 |
| 128 | 16×32 | **60.25 KB** | ❌ 超过（需要自适应调整） |
| 256 | 32×64 | **240.5 KB** | ❌ 超过 |
| 256 | 16×32 | **116.25 KB** | ❌ 超过 |

**设计逻辑**：
- **head_dim <= 64**：使用较大 tile（32×64）
  - 内存占用：72.5 KB（超过限制，但代码会自适应调整）
  - 实际上会被调整为更小的 tile
  
- **head_dim > 64 且 < 128**：使用中等 tile（32×64）
  - 尝试使用较大 tile，如果超过限制会自动调整
  
- **head_dim >= 128**：使用较小 tile（16×32）
  - **必须减小 tile**，因为 head_dim 增大导致 Q/K/V 和 O accumulator 的内存线性增长
  - 即使 16×32 也可能超过 48KB，需要进一步自适应调整到 4×8

**关键理解**：
- **head_dim 增大 → Q/K/V 和 O 的内存线性增长**
- **Scores 的内存与 head_dim 无关**（只与 tile 大小有关）
- **为了不超过 48KB 限制，必须减小 tile 大小**

**自适应调整机制**（第 400-406 行）：
```cpp
if (shared_mem_size > max_shared_mem) {
    block_size_q = 4;
    block_size_kv = 8;
    shared_mem_size = smem_needed(block_size_q, block_size_kv);
}
```
如果初始选择的 tile 超过 48KB，会自动调整为最小的 tile（4×8）。

## 四、重标定的数值误差

### 4.1 误差来源

每次重标定都会引入误差：
1. **exp 计算误差**：`alpha = exp(m_old - m_new)` 的浮点误差
2. **乘法误差**：`alpha * o_old` 的舍入误差
3. **累积误差**：多次重标定会累积误差

### 4.2 误差累积示例

```
假设有 10 次重标定，每次引入 1e-6 的相对误差：

总误差 ≈ 10 × 1e-6 = 1e-5  ← 接近 float 路径的容差极限！
```

这就是为什么：
- **Float 路径需要更大的 tile**（减少重标定次数）
- **使用 double 作为中间量**（减少每次重标定的误差）
- **使用 Kahan Summation**（减少累加误差）

### 4.3 优化策略

代码中的优化：
1. **更大的 tile**：减少重标定次数
2. **Double 中间量**：提高重标定精度
3. **Kahan Summation**：减少累加误差
4. **FMA 优化**：减少点积计算误差

## 五、总结

### 5.1 Tile 大小的作用

| Tile 大小 | 重标定次数 | 内存占用 | 数值精度 | 适用场景 |
|----------|-----------|---------|---------|---------|
| 小 (8×16) | 多 | 少 | 较低 | Half，内存受限 |
| 中 (16×32) | 中 | 中 | 中等 | Half，大 head_dim |
| 大 (32×64) | 少 | 多 | 较高 | Float，精度要求高 |

### 5.2 重标定的重要性

- **重标定是 Online Softmax 的核心**：允许分块计算而不存储完整矩阵
- **重标定次数直接影响精度**：次数越少，误差越小
- **需要在精度和内存之间平衡**：更大的 tile 提高精度但占用更多内存

### 5.3 设计原则

1. **精度优先**：Float 路径使用更大的 tile
2. **内存效率**：Half 路径使用较小的 tile
3. **自适应调整**：根据 head_dim 和内存限制动态调整
4. **数值稳定**：使用 double 中间量和 Kahan Summation

---

**参考代码位置**：
- Tile 大小选择：`src/kernels.cu:390-399`
- 重标定逻辑：`src/kernels.cu:229-292`
- 内存计算：`src/kernels.cu:374-387`
