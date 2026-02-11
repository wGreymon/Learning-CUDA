# IDE 配置说明

## 问题描述

如果遇到以下问题：
- 无法跳转到标准库函数（如 `cudaMalloc`, `fmaf` 等）
- 编辑器显示大量红色波浪线报错
- IntelliSense 无法识别 CUDA 类型和函数

这通常是因为 IDE 没有正确配置 CUDA 头文件路径和编译选项。

## 已创建的配置文件

### 1. `.vscode/c_cpp_properties.json`
- 配置 C/C++ IntelliSense
- 包含 CUDA 头文件路径
- 设置编译选项和宏定义

### 2. `.vscode/settings.json`
- 配置文件关联（`.cu` → `cuda-cpp`）
- 设置默认的包含路径

### 3. `compile_commands.json`
- 用于 clangd 和其他 LSP 服务器
- 提供准确的编译命令信息

### 4. `.clangd`
- 如果使用 clangd 作为 LSP 服务器
- 配置编译标志和诊断选项

## 解决步骤

### 方法 1：重启编辑器（推荐）

1. **完全关闭 Cursor/VSCode**
2. **重新打开项目**
3. **等待 IntelliSense 索引完成**（右下角会显示索引进度）

### 方法 2：重新加载窗口

在 Cursor/VSCode 中：
- 按 `Ctrl+Shift+P` (Linux/Windows) 或 `Cmd+Shift+P` (Mac)
- 输入 "Reload Window"
- 选择 "Developer: Reload Window"

### 方法 3：检查 C/C++ 扩展

确保已安装以下扩展之一：
- **C/C++** (Microsoft) - 用于 IntelliSense
- **clangd** (LLVM) - 替代方案，通常更准确

### 方法 4：手动触发 IntelliSense 更新

1. 按 `Ctrl+Shift+P`
2. 输入 "C/C++: Reset IntelliSense Database"
3. 选择该命令

## 验证配置

### 检查 1：查看包含路径

在 `src/kernels.cu` 中，将光标放在 `#include <cuda_fp16.h>` 上：
- 应该能够 `Ctrl+Click` 跳转到文件
- 不应该显示红色波浪线

### 检查 2：查看标准库函数

将光标放在 `cudaMalloc` 上：
- 应该能够 `F12` 或 `Ctrl+Click` 跳转到定义
- 应该显示函数签名和文档

### 检查 3：查看 CUDA 类型

将光标放在 `half` 类型上：
- 应该能够跳转到 `cuda_fp16.h` 中的定义
- 不应该显示 "未定义类型" 错误

## 常见问题

### Q: 仍然无法跳转？

**A:** 尝试以下方法：
1. 检查 CUDA 路径是否正确：`ls /usr/local/cuda-13.1/include`
2. 如果 CUDA 安装在别处，修改 `.vscode/c_cpp_properties.json` 中的路径
3. 检查是否有多个 C/C++ 扩展冲突，禁用不需要的

### Q: 仍然有大量报错？

**A:** 
1. 确保 `compile_commands.json` 存在且格式正确
2. 如果使用 clangd，确保 `.clangd` 文件存在
3. 检查 C/C++ 扩展的设置，确保使用正确的配置

### Q: IntelliSense 很慢？

**A:**
1. 排除不必要的文件夹（在 `c_cpp_properties.json` 的 `browse.path` 中）
2. 减少 `browse.limitSymbolsToIncludedHeaders` 的范围
3. 考虑使用 clangd 替代默认的 IntelliSense

## CUDA 路径配置

如果 CUDA 安装在不同位置，需要修改以下文件中的路径：

1. `.vscode/c_cpp_properties.json` - 修改所有 `/usr/local/cuda-13.1` 路径
2. `compile_commands.json` - 修改 `-I` 参数中的路径
3. `.clangd` - 修改 `Add` 部分的 `-I` 参数

查找 CUDA 路径：
```bash
which nvcc
# 输出类似：/usr/local/cuda-13.1/bin/nvcc
# 则 CUDA 路径为：/usr/local/cuda-13.1
```

## 推荐扩展

- **C/C++** (Microsoft) - 基础 IntelliSense
- **C/C++ Extension Pack** - 包含多个有用的扩展
- **clangd** (LLVM) - 更准确的代码分析（可选，与 C/C++ 扩展二选一）

## 注意事项

- 如果同时安装了 C/C++ 和 clangd，可能会冲突，建议只使用一个
- `compile_commands.json` 可以通过 `bear` 工具自动生成（如果安装了）
- 某些 CUDA 特定的语法（如 `__global__`）可能仍然会显示警告，这是正常的
