// MLIR Transform Dialect Scheduling Script for Apple M4
// High-Performance 2-Tier Caching + Vector Contraction Architecture

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // 1. 匹配锁定 Payload IR 中的算子句柄
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %dequant = transform.structured.match ops{["linalg.generic"]} attributes {filter_attr_name = "dequant"} in %root : (!transform.any_op) -> !transform.any_op

    // 2. 第一级分块：L1D 缓存级分块 [64, 64, 128] (精准契合 M4 单核 128KB L1D)
    %tiled_matmul_l1, %loops_l1:3 = transform.structured.tile_using_for %matmul tile_sizes [64, 64, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 3. 算子融合 (Fusion)：将反量化生产者融合进 L1 分块循环，消除中间 DRAM 读写！
    %fused_dequant, %fused_loop = transform.structured.fuse_into_containing_op %dequant into %loops_l1#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. 第二级分块：寄存器循环级分块 [8, 16, 8] (生成紧凑无气泡的硬件寄存器遍历循环)
    %tiled_matmul_reg, %loops_reg:3 = transform.structured.tile_using_for %tiled_matmul_l1 tile_sizes [8, 16, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 5. 向量化：以 [8, 16, 8] 向量切片直接下沉为 128-bit NEON 原生向量指令
    transform.structured.vectorize %tiled_matmul_reg vector_sizes [8, 16, 8] : !transform.any_op

    transform.yield
  }
}
