// Schedule S5: Wide Vector Register Schedule ([16, 16, 8] with Double SIMD Accumulator)
// Targets M4 4-Wide NEON Execution Ports with Maximized FMA Instruction Density

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // 1. 匹配算子
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %dequant = transform.structured.match ops{["linalg.generic"]} attributes {filter_attr_name = "dequant"} in %root : (!transform.any_op) -> !transform.any_op

    // 2. 第一级分块：L1D 缓存级分块 [64, 128, 128]
    %tiled_matmul_l1, %loops_l1:3 = transform.structured.tile_using_for %matmul tile_sizes [64, 128, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 3. 算子融合 (Fusion)
    %fused_dequant, %fused_loop = transform.structured.fuse_into_containing_op %dequant into %loops_l1#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. 第二级分块：超宽向量寄存器切片 [16, 16, 8]
    %tiled_matmul_reg, %loops_reg:3 = transform.structured.tile_using_for %tiled_matmul_l1 tile_sizes [16, 16, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 5. 向量化
    transform.structured.vectorize %tiled_matmul_reg vector_sizes [16, 16, 8] : !transform.any_op

    transform.yield
  }
}
