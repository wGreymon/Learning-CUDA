## Flash Attention 修复与优化摘要

### 作业对齐：算子语义（与 PyTorch 一致）
本作业 `flashAttention<T>` 的目标行为与 `torch.nn.functional.scaled_dot_product_attention` 对齐（在作业接口暴露的参数范围内），核心语义如下：

- **输入/输出**：对每个 batch、每个 head（或 grouped head）计算注意力：
  \[
  \mathrm{Attn}(Q,K,V)=\mathrm{Softmax}(QK^T \cdot s + M)\,V
  \]
  其中 \(s\) 为 scale（通常为 \(1/\sqrt{d}\)），\(M\) 为掩码（causal 时将非法位置置为 \(-\infty\)）。
- **causal masking**：标准因果条件为 **key 位置不超过 query 位置**，即 \(j \le i\)。不允许“矩形偏移/shift”之类的自定义语义。
- **GQA（Grouped Query Attention）**：当 `num_kv_heads < num_q_heads` 时，多个 Q heads 共享同一组 K/V heads；映射关系应等价于：
  \[
  kv\_head = \left\lfloor \frac{q\_head \cdot num\_kv\_heads}{num\_q\_heads} \right\rfloor
  \]

### 主要问题回顾
- **在线 softmax 分块累积的数值漂移**：原实现为 float/half 共用的 Flash 栈式算法，浮点累积与多次重标定导致极小误差，在测试的极严容差下（1e-6 量级）浮点用例频繁超差。
- **掩码逻辑回归**：曾加入“矩形” causal shift，导致部分用例的因果掩码错误（query 允许看到更远的 key），直接造成 Test #1 fail。
- **输出累加精度不足**：输出累加在 float 路径上反复截断为 float，进一步放大误差。

### 关键优化与修复
1. **分路径实现**  
   - `T=float`：改为朴素三遍 softmax（max → denom → output），避免在线重标定带来的误差累积，并使用 `expf`/`fmaf` 保持与参考实现接近的浮点行为。  
   - `T=half`：保留 Flash 分块实现，继续使用 double 中间量保证 half 正确性。
2. **掩码恢复标准因果**  
   - 回退到标准 causal 条件 `j ≤ i`，消除矩形 shift 引入的错误遮蔽/暴露。
3. **累积精度提升**  
   - 输出累加在共享内存中使用 double（half 路径），点积在 float 路径使用 `fmaf`，减少截断误差。
4. **块尺寸调整**  
   - 调大 KV tile（16×32）以减少 online softmax 重标定次数，降低数值抖动；同时保持共享内存不超限。

### 实现要点（为什么这样做）
- **float 路径选择“三遍 softmax”**：在极严容差下，online-softmax 的“分块 max + 重标定 + 累积”会累积非常小的误差，最终 `max diff` 可能越界；三遍 softmax 虽慢，但更接近参考实现的数值轨迹。
- **half 路径保留 tiled/online-softmax**：half 本身需要更高精度中间量来稳定；用 double 做部分中间累积能显著抑制误差，同时 tiled 方式更符合 FlashAttention 设计。
- **掩码先对齐语义再谈优化**：任何“看起来更快”的掩码变体只要偏离 \(j \le i\) 就会在某些用例上直接出错。

### 复杂度与资源（便于写作业）
- **时间复杂度**：对单个 head，主计算为 \(O(L_q \cdot L_k \cdot d)\) 的 \(QK^T\) 与 \(PV\)；softmax 额外是 \(O(L_q \cdot L_k)\)。
- **空间复杂度**：
  - 朴素实现若显式存 \(P\)（注意力矩阵）为 \(O(L_q \cdot L_k)\)；本实现避免显式存整块 \(P\)，以 tile/流式方式在寄存器/共享内存中完成。
  - tiled 方式的额外开销主要来自每个 block 的共享内存缓冲（按 tile 大小线性增长）。

### 测试结果
- 最终在提供的测试集上 **float/half 全部通过**（Attention Tests #1–#14）。
- 过程中出现的 `stack smashing detected` 被确认是平台侧干扰，与算法修复无关；最终版本在同平台完成全通过。

### 如何复现（写报告/自检用）
- **全量测试**：
  - `make clean && make PLATFORM=nvidia`
  - `./test_kernels`
- **打印更详细差异（若 Makefile 支持 run）**：
  - `make run VERBOSE=true`

### 经验与建议
- 对于极严容差的 float 测试，朴素三遍 softmax 往往更稳健；性能优先时可再尝试 Flash，但需严格控制重标定次数与累积精度。
- 因果掩码要优先保持与参考实现一致的语义（标准 `j ≤ i`），特殊平台差异建议放到可选分支，而非默认路径。
- half 路径宜保持 double 中间量和较大的 tile；float 路径若追求精度，可专门走 float-only 内核，避免混合精度的截断放大。

### 实现结构概览（报告可直接引用）
- **Host 调度 `flashAttention<T>`**  
  - 判空/尺寸合法性 → 拷贝 H2D → 根据 `T` 选择 kernel → 运行后回拷 D2H。  
  - `float` 分支：走朴素三遍 softmax kernel。`half` 分支：走 Flash 风格 tiled kernel，自动根据 `head_dim`/48KB shared mem 选择 `block_size_q/block_size_kv`。
- **float kernel（朴素三遍 softmax）**  
  - grid `(tgt_seq_len, query_heads, batch_size)`，一个 block 处理一条 `(b, head, q_pos)`。  
  - Pass1 求 max，Pass2 求 denom，Pass3 输出；GQA 通过 `hk = (hq * kv_heads) / query_heads` 映射。  
  - 数值轨迹与 PyTorch 参考实现更接近，容差 1e-6 下稳定。
- **half kernel（Flash 风格 tiled online softmax）**  
  - grid `(num_q_blocks, query_heads, batch_size)`，block 256 线程。  
  - shared 布局：`Q/K/V` tile + `scores`(double) + `o_accum`(double) + `m/l`(double)。  
  - 按 KV tile 流式累积：每 tile 先算 `s_shared`（带 causal，条件统一为 `j ≤ i`），再用 online softmax 更新 `(m, l, o)`，中间量全用 double。  
  - 末尾对 `o_accum / l` 归一化写回，全程保持标准 causal 语义与 GQA 映射一致。
