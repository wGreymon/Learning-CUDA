# 为什么 head_dim 越大，tile 越小？

## 问题

代码中：
```cpp
block_size_q  = (head_dim >= 128) ? 16 : ((head_dim > 64) ? 32 : 32);
block_size_kv = (head_dim >= 128) ? 32 : ((head_dim > 64) ? 64 : 64);
```

**为什么 head_dim 越大，block_size 反而越小？**

## 答案：内存限制

### 1. Shared Memory 限制

CUDA 的 shared memory 通常限制为 **48 KB**（某些架构可能更多，但这里是保守估计）。

### 2. 内存占用公式

让我们分解内存占用：

```cpp
总内存 = Q_tile + K_tile + V_tile + Scores + O_accumulator + M_L

其中：
Q_tile = block_size_q × head_dim × sizeof(T)
K_tile = block_size_kv × head_dim × sizeof(T)
V_tile = block_size_kv × head_dim × sizeof(T)
Scores = block_size_q × block_size_kv × sizeof(double)
O_accumulator = block_size_q × head_dim × sizeof(double)
M_L = 2 × block_size_q × sizeof(double)
```

**关键观察**：
- **Q/K/V 和 O 的内存与 `head_dim` 成正比**：`O(head_dim)`
- **Scores 的内存与 `head_dim` 无关**：`O(1)` 相对于 head_dim

### 3. 实际计算示例

#### 情况 1：head_dim = 64（float 类型）

**Tile 32×64**：
```
Q: 32 × 64 × 4 = 8,192 bytes
K: 64 × 64 × 4 = 16,384 bytes
V: 64 × 64 × 4 = 16,384 bytes
Q/K/V 小计: 40,960 bytes

对齐后: 40,960 bytes

Scores: 32 × 64 × 8 = 16,384 bytes
对齐后: 16,384 bytes

O: 32 × 64 × 8 = 16,384 bytes
对齐后: 16,384 bytes

M/L: 2 × 32 × 8 = 512 bytes
对齐后: 512 bytes

总计: 40,960 + 16,384 + 16,384 + 512 = 74,240 bytes ≈ 72.5 KB ❌ 超过 48KB！
```

**Tile 16×32**：
```
Q: 16 × 64 × 4 = 4,096 bytes
K: 32 × 64 × 4 = 8,192 bytes
V: 32 × 64 × 4 = 8,192 bytes
Q/K/V 小计: 20,480 bytes

对齐后: 20,480 bytes

Scores: 16 × 32 × 8 = 4,096 bytes
对齐后: 4,096 bytes

O: 16 × 64 × 8 = 8,192 bytes
对齐后: 8,192 bytes

M/L: 2 × 16 × 8 = 256 bytes
对齐后: 256 bytes

总计: 20,480 + 4,096 + 8,192 + 256 = 33,024 bytes ≈ 32.25 KB ✅ 安全
```

#### 情况 2：head_dim = 128（float 类型）

**Tile 32×64**：
```
Q: 32 × 128 × 4 = 16,384 bytes
K: 64 × 128 × 4 = 32,768 bytes
V: 64 × 128 × 4 = 32,768 bytes
Q/K/V 小计: 81,920 bytes

对齐后: 81,920 bytes

Scores: 32 × 64 × 8 = 16,384 bytes（不变！）
对齐后: 16,384 bytes

O: 32 × 128 × 8 = 32,768 bytes（翻倍！）
对齐后: 32,768 bytes

M/L: 2 × 32 × 8 = 512 bytes（不变）
对齐后: 512 bytes

总计: 81,920 + 16,384 + 32,768 + 512 = 131,584 bytes ≈ 128.5 KB ❌ 超过 48KB！
```

**Tile 16×32**：
```
Q: 16 × 128 × 4 = 8,192 bytes
K: 32 × 128 × 4 = 16,384 bytes
V: 32 × 128 × 4 = 16,384 bytes
Q/K/V 小计: 40,960 bytes

对齐后: 40,960 bytes

Scores: 16 × 32 × 8 = 4,096 bytes
对齐后: 4,096 bytes

O: 16 × 128 × 8 = 16,384 bytes
对齐后: 16,384 bytes

M/L: 2 × 16 × 8 = 256 bytes
对齐后: 256 bytes

总计: 40,960 + 4,096 + 16,384 + 256 = 61,696 bytes ≈ 60.25 KB ❌ 仍然超过 48KB！
```

### 4. 内存增长分析

**当 head_dim 从 64 增加到 128（翻倍）时**：

| 组件 | Tile 32×64 | Tile 16×32 |
|------|-----------|-----------|
| Q/K/V | 40,960 → 81,920 (+100%) | 20,480 → 40,960 (+100%) |
| Scores | 16,384 → 16,384 (不变) | 4,096 → 4,096 (不变) |
| O | 16,384 → 32,768 (+100%) | 8,192 → 16,384 (+100%) |
| M/L | 512 → 512 (不变) | 256 → 256 (不变) |
| **总计** | **72.5 KB → 128.5 KB** | **32.25 KB → 60.25 KB** |

**关键发现**：
- Q/K/V 和 O 的内存**线性增长**（与 head_dim 成正比）
- Scores 和 M/L 的内存**不变**（与 head_dim 无关）
- **head_dim 翻倍 → Q/K/V 和 O 翻倍 → 总内存大幅增加**

### 5. 设计逻辑

```cpp
if constexpr (std::is_same_v<T, float>) {
    block_size_q  = (head_dim >= 128) ? 16 : ((head_dim > 64) ? 32 : 32);
    block_size_kv = (head_dim >= 128) ? 32 : ((head_dim > 64) ? 64 : 64);
}
```

**设计思路**：
1. **head_dim <= 64**：尝试使用较大 tile（32×64）
   - 虽然可能超过 48KB，但代码会自适应调整
   - 优先考虑精度（减少重标定次数）

2. **head_dim > 64 且 < 128**：继续使用较大 tile（32×64）
   - 如果超过限制，自适应调整

3. **head_dim >= 128**：**必须减小 tile**（16×32）
   - Q/K/V 和 O 的内存已经很大
   - 即使 16×32 也可能超过 48KB，需要进一步调整

### 6. 自适应调整机制

代码中的安全检查（第 400-406 行）：
```cpp
size_t shared_mem_size = smem_needed(block_size_q, block_size_kv);
const size_t max_shared_mem = 48 * 1024;
if (shared_mem_size > max_shared_mem) {
    block_size_q = 4;
    block_size_kv = 8;
    shared_mem_size = smem_needed(block_size_q, block_size_kv);
}
```

**作用**：
- 如果初始选择的 tile 超过 48KB
- 自动调整为最小的 tile（4×8）
- 确保不会超过 shared memory 限制

### 7. 总结

**为什么 head_dim 越大，tile 越小？**

1. **内存限制**：Shared memory 限制为 48KB
2. **线性增长**：Q/K/V 和 O 的内存与 head_dim 成正比
3. **必须减小**：head_dim 增大时，必须减小 tile 才能满足内存限制
4. **精度权衡**：虽然减小 tile 会增加重标定次数，但这是内存限制下的必要权衡

**公式总结**：
```
内存占用 = f(block_size_q, block_size_kv, head_dim)
         = O(block_size × head_dim) + O(block_size²)

当 head_dim ↑ 时，为了保持 内存占用 < 48KB
必须使 block_size ↓
```

---

**参考**：
- 内存计算函数：`src/kernels.cu:374-387`
- Tile 大小选择：`src/kernels.cu:390-399`
- 自适应调整：`src/kernels.cu:400-406`
