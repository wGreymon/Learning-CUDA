# CUDA Shared Memory 限制：48KB 的来源

## 问题

代码中：
```cpp
const size_t max_shared_mem = 48 * 1024;  // 48 KB
```

**这个值是怎么来的？**

## 答案：CUDA 架构的默认限制

### 1. 48KB 是每个 Thread Block 的默认限制

**48 KB** 是 CUDA 中**每个 thread block 可以使用的 shared memory 的默认上限**。

### 2. CUDA 架构的 Shared Memory 限制

#### 2.1 不同计算能力的限制

| 计算能力 | 架构 | 每个 Thread Block 默认限制 | 每个 SM 总容量 | 每个 SM 最大限制 |
|---------|------|-------------------------|--------------|----------------|
| 6.x | Pascal | 48 KB | 48 KB | 48 KB |
| 7.0 | Volta | 48 KB (默认) | 96 KB | 96 KB |
| 7.5 | Turing | 48 KB (默认) | 64 KB | 64 KB |
| 8.0 | Ampere (A100) | 48 KB (默认) | 164 KB | 163 KB |
| 8.6 | Ampere (RTX 30xx) | 48 KB (默认) | 100 KB | 99 KB |
| 8.9 | Ada/Ampere+ | 48 KB (默认) | 100 KB+ | 99 KB+ |

**关键点**：
- **48 KB 是默认限制**，适用于大多数架构
- 某些架构（如 Volta、Ampere）支持更多，但需要**显式启用**
- 代码使用 48 KB 作为**保守的通用限制**，确保在所有 GPU 上都能运行

#### 2.2 你的 GPU 信息

根据 `nvidia-smi` 查询，你的 GPU 计算能力是 **8.9**（Ada/Ampere+ 架构），理论上可以支持：
- 每个 thread block：最多 99 KB（需要显式启用）
- 每个 SM：100 KB+

但代码仍然使用 **48 KB** 作为限制，原因：
1. **兼容性**：确保代码在所有 GPU 上都能运行
2. **简单性**：不需要动态检测和配置
3. **保守策略**：避免在某些配置下失败

### 3. Shared Memory 的分配方式

#### 3.1 静态 Shared Memory

```cpp
__shared__ float shared_array[1024];  // 编译时分配
```

#### 3.2 动态 Shared Memory（代码中使用的方式）

```cpp
// 在 kernel 启动时指定大小
flash_attention_kernel<<<grid, block, shared_mem_size>>>(
    // ...
);

// 在 kernel 内部使用
extern __shared__ char shared_mem[];
```

**动态 shared memory 的限制**：
- 默认：**48 KB per thread block**
- 超过 48 KB：需要显式设置 `cudaFuncSetAttribute`

### 4. 为什么选择 48 KB？

#### 4.1 通用兼容性

```cpp
const size_t max_shared_mem = 48 * 1024;  // 48 KB
```

**选择 48 KB 的原因**：
1. **最低公共分母**：所有现代 CUDA 架构都支持至少 48 KB
2. **无需特殊配置**：不需要调用 `cudaFuncSetAttribute`
3. **简单可靠**：避免架构特定的代码路径

#### 4.2 实际使用情况

代码中的内存计算：
```cpp
auto smem_needed = [&](int bq, int bkv) {
    // 计算所需内存
    // Q/K/V + Scores + O accumulator + M/L
    return total_size;
};

if (shared_mem_size > max_shared_mem) {
    // 如果超过 48 KB，减小 tile 大小
    block_size_q = 4;
    block_size_kv = 8;
}
```

**实际内存占用示例**：
- Tile 32×64, head_dim=64: **72.5 KB** ❌ 超过 48 KB
- Tile 16×32, head_dim=64: **32.25 KB** ✅ 安全
- Tile 4×8, head_dim=128: **约 10 KB** ✅ 安全

### 5. 如何突破 48 KB 限制？

如果需要使用更多 shared memory（例如在 A100 上），可以：

#### 方法 1：显式设置更大的限制

```cpp
// 查询 GPU 能力
int max_shared_mem;
cudaDeviceGetAttribute(&max_shared_mem, 
                      cudaDevAttrMaxSharedMemoryPerBlock, 
                      device_id);

// 设置 kernel 属性（如果支持）
cudaFuncSetAttribute(flash_attention_kernel<T>, 
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 
                    max_shared_mem);
```

#### 方法 2：使用更大的限制值

```cpp
// 根据计算能力选择
int compute_capability = ...;  // 查询 GPU
size_t max_shared_mem;
if (compute_capability >= 80) {
    max_shared_mem = 99 * 1024;  // Ampere: 99 KB
} else if (compute_capability >= 70) {
    max_shared_mem = 96 * 1024;  // Volta: 96 KB
} else {
    max_shared_mem = 48 * 1024;  // 默认: 48 KB
}
```

**但当前代码选择简单方案**：
- 使用固定的 48 KB 限制
- 通过减小 tile 大小来适应
- 确保在所有 GPU 上都能运行

### 6. 48 KB 的限制来源

#### 6.1 历史原因

- **早期架构**（如 Fermi, Kepler）：48 KB 是硬限制
- **后续架构**：虽然支持更多，但**默认仍然是 48 KB**，需要显式启用

#### 6.2 CUDA 编程指南

根据 NVIDIA CUDA 编程指南：
- **默认行为**：每个 thread block 可以使用最多 48 KB shared memory
- **超出限制**：需要使用 `cudaFuncSetAttribute` 显式请求更多

#### 6.3 实际限制

即使 GPU 支持更多 shared memory，也有其他限制：
1. **每个 SM 的总容量**：需要分配给多个 thread block
2. **寄存器使用**：影响每个 SM 可以运行的 thread block 数量
3. **资源竞争**：多个 thread block 共享同一个 SM 的资源

### 7. 代码中的使用

```cpp
const size_t max_shared_mem = 48 * 1024;  // 48 KB

if (shared_mem_size > max_shared_mem) {
    // 自适应调整：减小 tile 大小
    block_size_q = 4;
    block_size_kv = 8;
    shared_mem_size = smem_needed(block_size_q, block_size_kv);
}
```

**设计思路**：
1. **保守限制**：使用 48 KB 确保兼容性
2. **自适应调整**：如果超过限制，自动减小 tile
3. **保证运行**：确保在任何 GPU 上都能成功启动 kernel

### 8. 总结

**48 KB 的来源**：
1. **CUDA 架构的默认限制**：每个 thread block 默认可以使用 48 KB shared memory
2. **通用兼容性**：所有现代 GPU 都支持至少 48 KB
3. **保守策略**：确保代码在所有 GPU 上都能运行
4. **简单可靠**：不需要架构特定的配置代码

**为什么不用更大的值**：
- 虽然你的 GPU（计算能力 8.9）支持更多，但：
  - 需要显式配置（`cudaFuncSetAttribute`）
  - 增加代码复杂度
  - 可能在某些 GPU 上失败
  - 当前设计通过减小 tile 已经足够

**实际效果**：
- 48 KB 限制是合理的
- 通过自适应调整 tile 大小，可以适应各种情况
- 代码简洁且可靠

---

**参考**：
- NVIDIA CUDA Programming Guide: Compute Capabilities
- CUDA Best Practices Guide: Shared Memory
- 代码位置：`src/kernels.cu:402`
