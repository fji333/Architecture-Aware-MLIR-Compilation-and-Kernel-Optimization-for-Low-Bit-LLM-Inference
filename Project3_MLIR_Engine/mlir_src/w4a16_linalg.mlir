// W4A16 GEMM Linalg Payload IR Definition
// Apple M4 Optimization Battleground

#map_packed = affine_map<(d0, d1) -> (d0, d1 floordiv 2)>
#map_scale  = affine_map<(d0, d1) -> (d0 floordiv 128, d1)>
#map_zero   = affine_map<(d0, d1) -> (d0 floordiv 128, d1)>
#map_out    = affine_map<(d0, d1) -> (d0, d1)>
#map_bias   = affine_map<(d0, d1) -> (d1)>

module {
  func.func @w4a16_gemm_fused(
      %A: tensor<?x?xf16>,            // M x K
      %B_packed: tensor<?x?xi8>,      // K x (N/2)
      %scales: tensor<?x?xf16>,       // (K/128) x N
      %zeros: tensor<?x?xf16>,        // (K/128) x N
      %bias: tensor<?xf16>,           // N
      %C_init: tensor<?x?xf16>        // M x N (Accumulator)
  ) -> tensor<?x?xf16> attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0_K = tensor.dim %B_packed, %c0 : tensor<?x?xi8>
    %d1_N_half = tensor.dim %B_packed, %c1 : tensor<?x?xi8>
    %c2 = arith.constant 2 : index
    %d1_N = arith.muli %d1_N_half, %c2 : index

    // 1. 非对称子字节反量化解包算子 (linalg.generic)
    %B_init = tensor.empty(%d0_K, %d1_N) : tensor<?x?xf16>
    %c15_i8 = arith.constant 15 : i8
    %c4_i8  = arith.constant 4 : i8

    %B_dequant = linalg.generic {
      indexing_maps = [#map_packed, #map_scale, #map_zero, #map_out],
      iterator_types = ["parallel", "parallel"]
    } ins(%B_packed, %scales, %zeros : tensor<?x?xi8>, tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%B_init : tensor<?x?xf16>) attrs = {filter_attr_name = "dequant"} {
    ^bb0(%in_packed: i8, %scale: f16, %zero: f16, %out: f16):
      %col = linalg.index 1 : index
      %is_odd_idx = arith.andi %col, %c1 : index
      %is_odd = arith.index_cast %is_odd_idx : index to i1
      
      %val_even = arith.andi %in_packed, %c15_i8 : i8
      %shifted = arith.shrui %in_packed, %c4_i8 : i8
      %val_odd = arith.andi %shifted, %c15_i8 : i8
      
      %val_i8 = arith.select %is_odd, %val_odd, %val_even : i8
      
      %val_f16 = arith.sitofp %val_i8 : i8 to f16
      %sub = arith.subf %val_f16, %zero : f16
      %dequant = arith.mulf %sub, %scale : f16
      
      linalg.yield %dequant : f16
    } -> tensor<?x?xf16>

    // 2. 矩阵乘法收缩 (linalg.matmul)
    %C_matmul = linalg.matmul
      ins(%A, %B_dequant : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%C_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 3. 偏置加法融合 (linalg.generic)
    %C_final = linalg.generic {
      indexing_maps = [#map_out, #map_bias, #map_out],
      iterator_types = ["parallel", "parallel"]
    } ins(%C_matmul, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%C_init : tensor<?x?xf16>) attrs = {filter_attr_name = "bias_add"} {
    ^bb0(%val: f16, %b: f16, %out: f16):
      %add = arith.addf %val, %b : f16
      linalg.yield %add : f16
    } -> tensor<?x?xf16>

    return %C_final : tensor<?x?xf16>
  }
}
