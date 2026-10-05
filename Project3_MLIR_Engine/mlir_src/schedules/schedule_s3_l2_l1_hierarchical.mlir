// Schedule S3: 2-Tier Hierarchical Caching (L2 + L1) + Producer Fusion + Register Vectorization
// Exploits Apple M4 12MB Shared L2 and 128KB Private L1D Cache

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // 1. 匹配算子
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %dequant = transform.structured.match ops{["linalg.generic"]} attributes {filter_attr_name = "dequant"} in %root : (!transform.any_op) -> !transform.any_op

    // 2. 第一级分块：L2 缓存级分块 [256, 512, 0] (匹配 M4 12MB L2)
    %tiled_matmul_l2, %loops_l2:2 = transform.structured.tile_using_for %matmul tile_sizes [256, 512, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 3. 算子融合 (Fusion)：将反量化生产者融合进 L2 分块循环
    %fused_dequant, %fused_loop = transform.structured.fuse_into_containing_op %dequant into %loops_l2#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. 第二级分块：L1 缓存级分块 [64, 64, 128] (匹配 M4 128KB L1D)
    %tiled_matmul_l1, %loops_l1:3 = transform.structured.tile_using_for %tiled_matmul_l2 tile_sizes [64, 64, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 5. 第三级分块：寄存器循环级分块 [8, 16, 8]
    %tiled_matmul_reg, %loops_reg:3 = transform.structured.tile_using_for %tiled_matmul_l1 tile_sizes [8, 16, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 6. 向量化
    transform.structured.vectorize %tiled_matmul_reg vector_sizes [8, 16, 8] : !transform.any_op

    transform.yield
  }
}
