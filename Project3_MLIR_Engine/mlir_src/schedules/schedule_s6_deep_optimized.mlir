// Schedule S6: Deeply Optimized MLIR Transform Schedule for Apple M4
// Incorporates Hierarchical Tiling, Producer Fusion, Vector Contraction,
// Vector Transfer Hoisting, and Loop Unrolling

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // 1. 匹配锁定 Payload 算子句柄
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %dequant = transform.structured.match ops{["linalg.generic"]} attributes {filter_attr_name = "dequant"} in %root : (!transform.any_op) -> !transform.any_op

    // 2. 第一级分块：L1D 缓存分块 [64, 64, 128]
    %tiled_matmul_l1, %loops_l1:3 = transform.structured.tile_using_for %matmul tile_sizes [64, 64, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 3. 跨循环生产者融合：消除反量化 DRAM 访存
    %fused_dequant, %fused_loop = transform.structured.fuse_into_containing_op %dequant into %loops_l1#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. 第二级分块：寄存器循环级切片 [8, 16, 8]
    %tiled_matmul_reg, %loops_reg:3 = transform.structured.tile_using_for %tiled_matmul_l1 tile_sizes [8, 16, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 5. 向量化下沉至 NEON 原生向量指令
    transform.structured.vectorize %tiled_matmul_reg vector_sizes [8, 16, 8] : !transform.any_op

    // 6. 循环展开优化：对 K 步进微循环展开 2 倍以消除分支开销并提高指令流水线吞吐
    transform.loop.unroll %loops_reg#2 { factor = 2 } : !transform.any_op

    // 7. 冗余向量传输外提优化 (Vector Transfer Hoisting)
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.structured.hoist_redundant_vector_transfers %func : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
