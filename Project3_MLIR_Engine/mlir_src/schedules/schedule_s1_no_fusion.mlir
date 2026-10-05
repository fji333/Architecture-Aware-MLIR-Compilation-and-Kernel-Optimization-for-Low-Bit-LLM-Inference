// Schedule S1: Baseline Single-Tier Tiling without Operator Fusion
// Explores DRAM traffic baseline and separate kernel dispatch behavior

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // 1. 匹配锁定 Payload IR 中的算子句柄
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %dequant = transform.structured.match ops{["linalg.generic"]} attributes {filter_attr_name = "dequant"} in %root : (!transform.any_op) -> !transform.any_op

    // 2. 独立反量化循环分块 (无融合，中间结果写入 DRAM/临时 Buffer)
    %tiled_dequant, %loops_deq:2 = transform.structured.tile_using_for %dequant tile_sizes [64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 3. 独立矩阵乘法分块
    %tiled_matmul, %loops_mm:3 = transform.structured.tile_using_for %matmul tile_sizes [64, 64, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 4. 向量化
    transform.structured.vectorize %tiled_matmul vector_sizes [8, 16, 8] : !transform.any_op

    transform.yield
  }
}
