// Schedule S2: 1-Tier L1 Cache Tiling + Producer Fusion + Register Vectorization
// Focuses entirely on fitting the working set into Apple M4 128KB L1D cache

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // 1. 匹配算子
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %dequant = transform.structured.match ops{["linalg.generic"]} attributes {filter_attr_name = "dequant"} in %root : (!transform.any_op) -> !transform.any_op

    // 2. L1 缓存级分块 [64, 64, 128]
    %tiled_matmul_l1, %loops_l1:3 = transform.structured.tile_using_for %matmul tile_sizes [64, 64, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 3. 算子融合 (Fusion)：将反量化生产者融合进 L1 分块循环
    %fused_dequant, %fused_loop = transform.structured.fuse_into_containing_op %dequant into %loops_l1#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. 寄存器循环级分块 [8, 16, 8]
    %tiled_matmul_reg, %loops_reg:3 = transform.structured.tile_using_for %tiled_matmul_l1 tile_sizes [8, 16, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 5. 向量化叶子算子
    transform.structured.vectorize %tiled_matmul_reg vector_sizes [8, 16, 8] : !transform.any_op

    transform.yield
  }
}
