#map = affine_map<(d0, d1) -> (d0, d1 floordiv 2)>
#map1 = affine_map<(d0, d1) -> (d0 floordiv 128, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map4 = affine_map<(d0) -> (d0 - 1)>
#map5 = affine_map<(d0) -> (d0 floordiv 2)>
#map6 = affine_map<(d0) -> ((d0 - 1) floordiv 2)>
#map7 = affine_map<(d0) -> ((d0 - 1) floordiv 2 + 1)>
#map8 = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
#map9 = affine_map<(d0) -> (d0 floordiv 128)>
#map10 = affine_map<(d0) -> ((d0 - 1) floordiv 128)>
#map11 = affine_map<(d0) -> ((d0 - 1) floordiv 128 + 1)>
#map12 = affine_map<(d0)[s0] -> (d0 + s0)>
#map13 = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
#map14 = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map15 = affine_map<(d0, d1) -> (d0, 0, d1)>
#map16 = affine_map<(d0, d1) -> (0, d1, d0)>
#map17 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @w4a16_gemm_fused(%arg0: memref<?x?xf16, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xi8, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf16, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf16, strided<[?, ?], offset: ?>>, %arg4: memref<?xf16, strided<[?], offset: ?>>, %arg5: memref<?x?xf16, strided<[?, ?], offset: ?>>) -> memref<?x?xf16, strided<[?, ?], offset: ?>> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg1, %c0 : memref<?x?xi8, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xi8, strided<[?, ?], offset: ?>>
    %c2 = arith.constant 2 : index
    %0 = arith.muli %dim_0, %c2 : index
    %alloc = memref.alloc(%dim, %0) {alignment = 64 : i64} : memref<?x?xf16>
    %c15_i8 = arith.constant 15 : i8
    %c4_i8 = arith.constant 4 : i8
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2, %arg3 : memref<?x?xi8, strided<[?, ?], offset: ?>>, memref<?x?xf16, strided<[?, ?], offset: ?>>, memref<?x?xf16, strided<[?, ?], offset: ?>>) outs(%alloc : memref<?x?xf16>) attrs =  {filter_attr_name = "dequant"} {
    ^bb0(%in: i8, %in_66: f16, %in_67: f16, %out: f16):
      %14 = linalg.index 1 : index
      %15 = arith.andi %14, %c1 : index
      %16 = arith.index_cast %15 : index to i1
      %17 = arith.andi %in, %c15_i8 : i8
      %18 = arith.shrui %in, %c4_i8 : i8
      %19 = arith.andi %18, %c15_i8 : i8
      %20 = arith.select %16, %19, %17 : i8
      %21 = arith.sitofp %20 : i8 to f16
      %22 = arith.subf %21, %in_67 : f16
      %23 = arith.mulf %22, %in_66 : f16
      linalg.yield %23 : f16
    }
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %arg0, %c0_1 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg0, %c1_3 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c0_5 = arith.constant 0 : index
    %dim_6 = memref.dim %alloc, %c0_5 : memref<?x?xf16>
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %alloc, %c1_7 : memref<?x?xf16>
    %c0_9 = arith.constant 0 : index
    %dim_10 = memref.dim %arg5, %c0_9 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c1_11 = arith.constant 1 : index
    %dim_12 = memref.dim %arg5, %c1_11 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c0_13 = arith.constant 0 : index
    %c0_14 = arith.constant 0 : index
    %c0_15 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c64_16 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0_17 = arith.constant 0 : index
    %dim_18 = memref.dim %arg1, %c0_17 : memref<?x?xi8, strided<[?, ?], offset: ?>>
    %c1_19 = arith.constant 1 : index
    %dim_20 = memref.dim %arg1, %c1_19 : memref<?x?xi8, strided<[?, ?], offset: ?>>
    %c0_21 = arith.constant 0 : index
    %dim_22 = memref.dim %arg2, %c0_21 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c1_23 = arith.constant 1 : index
    %dim_24 = memref.dim %arg2, %c1_23 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c0_25 = arith.constant 0 : index
    %dim_26 = memref.dim %arg3, %c0_25 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c1_27 = arith.constant 1 : index
    %dim_28 = memref.dim %arg3, %c1_27 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c0_29 = arith.constant 0 : index
    %dim_30 = memref.dim %alloc, %c0_29 : memref<?x?xf16>
    %c1_31 = arith.constant 1 : index
    %dim_32 = memref.dim %alloc, %c1_31 : memref<?x?xf16>
    %c0_33 = arith.constant 0 : index
    %c1_34 = arith.constant 1 : index
    %c0_35 = arith.constant 0 : index
    %c1_36 = arith.constant 1 : index
    %c0_37 = arith.constant 0 : index
    %c1_38 = arith.constant 1 : index
    %c0_39 = arith.constant 0 : index
    %c0_40 = arith.constant 0 : index
    %c0_41 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c8_42 = arith.constant 8 : index
    %c1_43 = arith.constant 1 : index
    %1 = arith.subi %c8_42, %c1_43 : index
    %c2_44 = arith.constant 2 : index
    %2 = arith.muli %c8_42, %c2_44 : index
    %c0_45 = arith.constant 0 : index
    %c1_46 = arith.constant 1 : index
    %c1_47 = arith.constant 1 : index
    %c0_48 = arith.constant 0 : index
    %3 = ub.poison : f16
    %4 = ub.poison : f16
    %5 = ub.poison : f16
    %c0_49 = arith.constant 0 : index
    %c1_50 = arith.constant 1 : index
    %6 = arith.muli %c8_42, %c1_50 : index
    %c0_51 = arith.constant 0 : index
    %c1_52 = arith.constant 1 : index
    %c1_53 = arith.constant 1 : index
    %c0_54 = arith.constant 0 : index
    %7 = ub.poison : f16
    %8 = ub.poison : f16
    %9 = ub.poison : f16
    %c0_55 = arith.constant 0 : index
    %c0_56 = arith.constant 0 : index
    %c1_57 = arith.constant 1 : index
    %c1_58 = arith.constant 1 : index
    %c0_59 = arith.constant 0 : index
    %10 = ub.poison : f16
    %11 = ub.poison : f16
    %12 = ub.poison : f16
    %c0_60 = arith.constant 0 : index
    %c0_61 = arith.constant 0 : index
    %dim_62 = memref.dim %arg5, %c0_61 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %c1_63 = arith.constant 1 : index
    %dim_64 = memref.dim %arg5, %c1_63 : memref<?x?xf16, strided<[?, ?], offset: ?>>
    %alloc_65 = memref.alloc(%dim_62, %dim_64) {alignment = 64 : i64} : memref<?x?xf16>
    memref.copy %arg5, %alloc_65 : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16>
    %13 = scf.for %arg6 = %c0_13 to %dim_2 step %c64 iter_args(%arg7 = %alloc_65) -> (memref<?x?xf16>) {
      %14 = affine.min #map3(%arg6)[%dim_2]
      %15 = affine.apply #map4(%14)
      %16 = affine.apply #map4(%14)
      %17 = affine.apply #map4(%14)
      %18 = affine.apply #map4(%14)
      %19 = affine.apply #map4(%14)
      %20 = scf.for %arg8 = %c0_14 to %dim_8 step %c64_16 iter_args(%arg9 = %arg7) -> (memref<?x?xf16>) {
        %21 = affine.min #map3(%arg8)[%dim_8]
        %22 = affine.apply #map4(%21)
        %23 = affine.apply #map4(%21)
        %24 = affine.apply #map4(%21)
        %25 = affine.apply #map4(%21)
        %26 = affine.apply #map5(%arg8)
        %27 = affine.apply #map6(%21)
        %28 = affine.apply #map7(%21)
        %29 = affine.apply #map4(%21)
        %30 = affine.apply #map4(%21)
        %31 = affine.apply #map4(%21)
        %32 = affine.apply #map4(%21)
        %33 = affine.apply #map4(%21)
        %34 = scf.for %arg10 = %c0_15 to %dim_4 step %c128 iter_args(%arg11 = %arg9) -> (memref<?x?xf16>) {
          %35 = affine.min #map8(%arg10)[%dim_4]
          %36 = affine.apply #map4(%35)
          %37 = affine.apply #map4(%35)
          %38 = affine.apply #map4(%35)
          %subview = memref.subview %arg0[%arg6, %arg10] [%14, %35] [1, 1] : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
          %39 = affine.apply #map4(%35)
          %40 = affine.apply #map4(%35)
          %41 = affine.apply #map9(%arg10)
          %42 = affine.apply #map10(%35)
          %43 = affine.apply #map11(%35)
          %44 = affine.apply #map9(%arg10)
          %45 = affine.apply #map10(%35)
          %46 = affine.apply #map11(%35)
          %47 = affine.apply #map4(%35)
          %subview_66 = memref.subview %arg1[%arg10, %26] [%35, %28] [1, 1] : memref<?x?xi8, strided<[?, ?], offset: ?>> to memref<?x?xi8, strided<[?, ?], offset: ?>>
          %subview_67 = memref.subview %arg2[%41, %arg8] [%43, %21] [1, 1] : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
          %subview_68 = memref.subview %arg3[%44, %arg8] [%46, %21] [1, 1] : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
          %subview_69 = memref.subview %alloc[%arg10, %arg8] [%35, %21] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map1, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_66, %subview_67, %subview_68 : memref<?x?xi8, strided<[?, ?], offset: ?>>, memref<?x?xf16, strided<[?, ?], offset: ?>>, memref<?x?xf16, strided<[?, ?], offset: ?>>) outs(%subview_69 : memref<?x?xf16, strided<[?, 1], offset: ?>>) attrs =  {filter_attr_name = "dequant"} {
          ^bb0(%in: i8, %in_74: f16, %in_75: f16, %out: f16):
            %57 = linalg.index 1 : index
            %58 = affine.apply #map12(%arg8)[%57]
            %59 = arith.andi %58, %c1 : index
            %60 = arith.index_cast %59 : index to i1
            %61 = arith.andi %in, %c15_i8 : i8
            %62 = arith.shrui %in, %c4_i8 : i8
            %63 = arith.andi %62, %c15_i8 : i8
            %64 = arith.select %60, %63, %61 : i8
            %65 = arith.sitofp %64 : i8 to f16
            %66 = arith.subf %65, %in_75 : f16
            %67 = arith.mulf %66, %in_74 : f16
            linalg.yield %67 : f16
          }
          %subview_70 = memref.subview %arg11[%arg6, %arg8] [%14, %21] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %dim_71 = memref.dim %subview_69, %c0_35 : memref<?x?xf16, strided<[?, 1], offset: ?>>
          %dim_72 = memref.dim %subview_69, %c1_36 : memref<?x?xf16, strided<[?, 1], offset: ?>>
          %48 = arith.subi %35, %c0_41 : index
          %49 = arith.addi %48, %1 : index
          %50 = arith.divui %49, %c8_42 : index
          %51 = arith.remsi %50, %c2_44 : index
          %52 = arith.subi %50, %51 : index
          %53 = arith.muli %52, %c8_42 : index
          %54 = arith.addi %c0_41, %53 : index
          %55 = scf.for %arg12 = %c0_39 to %14 step %c8 iter_args(%arg13 = %subview_70) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
            %57 = affine.min #map13(%arg12)[%14]
            %58 = affine.apply #map4(%57)
            %59 = affine.apply #map4(%57)
            %60 = affine.apply #map4(%57)
            %61 = affine.apply #map4(%57)
            %62 = affine.apply #map4(%57)
            %63 = affine.min #map13(%arg12)[%14]
            %64 = affine.apply #map4(%63)
            %65 = affine.apply #map4(%63)
            %66 = affine.apply #map4(%63)
            %67 = affine.apply #map4(%63)
            %68 = affine.apply #map4(%63)
            %69 = affine.min #map13(%arg12)[%14]
            %70 = affine.apply #map4(%69)
            %71 = affine.apply #map4(%69)
            %72 = affine.apply #map4(%69)
            %73 = affine.apply #map4(%69)
            %74 = affine.apply #map4(%69)
            %75 = scf.for %arg14 = %c0_40 to %dim_72 step %c16 iter_args(%arg15 = %arg13) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
              %76 = affine.min #map14(%arg14)[%dim_72]
              %77 = affine.apply #map4(%76)
              %78 = affine.apply #map4(%76)
              %79 = affine.apply #map4(%76)
              %80 = affine.apply #map4(%76)
              %81 = affine.apply #map4(%76)
              %82 = affine.min #map14(%arg14)[%dim_72]
              %83 = affine.apply #map4(%82)
              %84 = affine.apply #map4(%82)
              %85 = affine.apply #map4(%82)
              %86 = affine.apply #map4(%82)
              %87 = affine.apply #map4(%82)
              %88 = scf.for %arg16 = %c0_41 to %54 step %2 iter_args(%arg17 = %arg15) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
                %96 = affine.min #map13(%arg16)[%35]
                %97 = affine.apply #map4(%96)
                %98 = affine.apply #map4(%96)
                %99 = affine.apply #map4(%96)
                %subview_74 = memref.subview %subview[%arg12, %arg16] [%57, %96] [1, 1] : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
                %subview_75 = memref.subview %subview_69[%arg16, %arg14] [%96, %76] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                %subview_76 = memref.subview %arg17[%arg12, %arg14] [%57, %76] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                %dim_77 = memref.dim %subview_74, %c0_45 : memref<?x?xf16, strided<[?, ?], offset: ?>>
                %dim_78 = memref.dim %subview_75, %c1_46 : memref<?x?xf16, strided<[?, 1], offset: ?>>
                %dim_79 = memref.dim %subview_74, %c1_47 : memref<?x?xf16, strided<[?, ?], offset: ?>>
                %100 = vector.create_mask %dim_77, %dim_79 : vector<8x8xi1>
                %101 = vector.mask %100 { vector.transfer_read %subview_74[%c0_48, %c0_48], %3 {in_bounds = [true, true, true], permutation_map = #map15} : memref<?x?xf16, strided<[?, ?], offset: ?>>, vector<8x16x8xf16> } : vector<8x8xi1> -> vector<8x16x8xf16>
                %102 = vector.create_mask %dim_79, %dim_78 : vector<8x16xi1>
                %103 = vector.mask %102 { vector.transfer_read %subview_75[%c0_48, %c0_48], %4 {in_bounds = [true, true, true], permutation_map = #map16} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8x16x8xf16> } : vector<8x16xi1> -> vector<8x16x8xf16>
                %104 = vector.create_mask %dim_77, %dim_78 : vector<8x16xi1>
                %105 = vector.mask %104 { vector.transfer_read %subview_76[%c0_48, %c0_48], %5 {in_bounds = [true, true]} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8x16xf16> } : vector<8x16xi1> -> vector<8x16xf16>
                %106 = arith.mulf %101, %103 : vector<8x16x8xf16>
                %107 = vector.create_mask %dim_77, %dim_78, %dim_79 : vector<8x16x8xi1>
                %108 = vector.mask %107 { vector.multi_reduction <add>, %106, %105 [2] : vector<8x16x8xf16> to vector<8x16xf16> } : vector<8x16x8xi1> -> vector<8x16xf16>
                vector.mask %104 { vector.transfer_write %108, %subview_76[%c0_49, %c0_49] {in_bounds = [true, true]} : vector<8x16xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>> } : vector<8x16xi1>
                %109 = affine.apply #map4(%96)
                %subview_80 = memref.subview %arg17[%arg12, %arg14] [%57, %76] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                memref.copy %subview_76, %subview_80 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                %110 = arith.addi %arg16, %6 : index
                %111 = affine.min #map13(%110)[%35]
                %112 = affine.apply #map4(%111)
                %113 = affine.apply #map4(%111)
                %114 = affine.apply #map4(%111)
                %subview_81 = memref.subview %subview[%arg12, %110] [%63, %111] [1, 1] : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
                %subview_82 = memref.subview %subview_69[%110, %arg14] [%111, %82] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                %subview_83 = memref.subview %arg17[%arg12, %arg14] [%63, %82] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                %dim_84 = memref.dim %subview_81, %c0_51 : memref<?x?xf16, strided<[?, ?], offset: ?>>
                %dim_85 = memref.dim %subview_82, %c1_52 : memref<?x?xf16, strided<[?, 1], offset: ?>>
                %dim_86 = memref.dim %subview_81, %c1_53 : memref<?x?xf16, strided<[?, ?], offset: ?>>
                %115 = vector.create_mask %dim_84, %dim_86 : vector<8x8xi1>
                %116 = vector.mask %115 { vector.transfer_read %subview_81[%c0_54, %c0_54], %7 {in_bounds = [true, true, true], permutation_map = #map15} : memref<?x?xf16, strided<[?, ?], offset: ?>>, vector<8x16x8xf16> } : vector<8x8xi1> -> vector<8x16x8xf16>
                %117 = vector.create_mask %dim_86, %dim_85 : vector<8x16xi1>
                %118 = vector.mask %117 { vector.transfer_read %subview_82[%c0_54, %c0_54], %8 {in_bounds = [true, true, true], permutation_map = #map16} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8x16x8xf16> } : vector<8x16xi1> -> vector<8x16x8xf16>
                %119 = vector.create_mask %dim_84, %dim_85 : vector<8x16xi1>
                %120 = vector.mask %119 { vector.transfer_read %subview_83[%c0_54, %c0_54], %9 {in_bounds = [true, true]} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8x16xf16> } : vector<8x16xi1> -> vector<8x16xf16>
                %121 = arith.mulf %116, %118 : vector<8x16x8xf16>
                %122 = vector.create_mask %dim_84, %dim_85, %dim_86 : vector<8x16x8xi1>
                %123 = vector.mask %122 { vector.multi_reduction <add>, %121, %120 [2] : vector<8x16x8xf16> to vector<8x16xf16> } : vector<8x16x8xi1> -> vector<8x16xf16>
                vector.mask %119 { vector.transfer_write %123, %subview_83[%c0_55, %c0_55] {in_bounds = [true, true]} : vector<8x16xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>> } : vector<8x16xi1>
                %124 = affine.apply #map4(%111)
                %subview_87 = memref.subview %arg17[%arg12, %arg14] [%63, %82] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                memref.copy %subview_83, %subview_87 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                scf.yield %arg17 : memref<?x?xf16, strided<[?, 1], offset: ?>>
              }
              %89 = affine.min #map14(%arg14)[%dim_72]
              %90 = affine.apply #map4(%89)
              %91 = affine.apply #map4(%89)
              %92 = affine.apply #map4(%89)
              %93 = affine.apply #map4(%89)
              %94 = affine.apply #map4(%89)
              %95 = scf.for %arg16 = %54 to %35 step %c8_42 iter_args(%arg17 = %88) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
                %96 = affine.min #map13(%arg16)[%35]
                %97 = affine.apply #map4(%96)
                %98 = affine.apply #map4(%96)
                %99 = affine.apply #map4(%96)
                %subview_74 = memref.subview %subview[%arg12, %arg16] [%69, %96] [1, 1] : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
                %subview_75 = memref.subview %subview_69[%arg16, %arg14] [%96, %89] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                %subview_76 = memref.subview %arg17[%arg12, %arg14] [%69, %89] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                %dim_77 = memref.dim %subview_74, %c0_56 : memref<?x?xf16, strided<[?, ?], offset: ?>>
                %dim_78 = memref.dim %subview_75, %c1_57 : memref<?x?xf16, strided<[?, 1], offset: ?>>
                %dim_79 = memref.dim %subview_74, %c1_58 : memref<?x?xf16, strided<[?, ?], offset: ?>>
                %100 = vector.create_mask %dim_77, %dim_79 : vector<8x8xi1>
                %101 = vector.mask %100 { vector.transfer_read %subview_74[%c0_59, %c0_59], %10 {in_bounds = [true, true, true], permutation_map = #map15} : memref<?x?xf16, strided<[?, ?], offset: ?>>, vector<8x16x8xf16> } : vector<8x8xi1> -> vector<8x16x8xf16>
                %102 = vector.create_mask %dim_79, %dim_78 : vector<8x16xi1>
                %103 = vector.mask %102 { vector.transfer_read %subview_75[%c0_59, %c0_59], %11 {in_bounds = [true, true, true], permutation_map = #map16} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8x16x8xf16> } : vector<8x16xi1> -> vector<8x16x8xf16>
                %104 = vector.create_mask %dim_77, %dim_78 : vector<8x16xi1>
                %105 = vector.mask %104 { vector.transfer_read %subview_76[%c0_59, %c0_59], %12 {in_bounds = [true, true]} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8x16xf16> } : vector<8x16xi1> -> vector<8x16xf16>
                %106 = arith.mulf %101, %103 : vector<8x16x8xf16>
                %107 = vector.create_mask %dim_77, %dim_78, %dim_79 : vector<8x16x8xi1>
                %108 = vector.mask %107 { vector.multi_reduction <add>, %106, %105 [2] : vector<8x16x8xf16> to vector<8x16xf16> } : vector<8x16x8xi1> -> vector<8x16xf16>
                vector.mask %104 { vector.transfer_write %108, %subview_76[%c0_60, %c0_60] {in_bounds = [true, true]} : vector<8x16xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>> } : vector<8x16xi1>
                %109 = affine.apply #map4(%96)
                %subview_80 = memref.subview %arg17[%arg12, %arg14] [%69, %89] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                memref.copy %subview_76, %subview_80 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
                scf.yield %arg17 : memref<?x?xf16, strided<[?, 1], offset: ?>>
              }
              scf.yield %95 : memref<?x?xf16, strided<[?, 1], offset: ?>>
            }
            scf.yield %75 : memref<?x?xf16, strided<[?, 1], offset: ?>>
          }
          %56 = affine.apply #map4(%35)
          %subview_73 = memref.subview %arg11[%arg6, %arg8] [%14, %21] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          memref.copy %55, %subview_73 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          scf.yield %arg11 : memref<?x?xf16>
        }
        scf.yield %34 : memref<?x?xf16>
      }
      scf.yield %20 : memref<?x?xf16>
    }
    linalg.generic {indexing_maps = [#map2, #map17, #map2], iterator_types = ["parallel", "parallel"]} ins(%13, %arg4 : memref<?x?xf16>, memref<?xf16, strided<[?], offset: ?>>) outs(%arg5 : memref<?x?xf16, strided<[?, ?], offset: ?>>) attrs =  {filter_attr_name = "bias_add"} {
    ^bb0(%in: f16, %in_66: f16, %out: f16):
      %14 = arith.addf %in, %in_66 : f16
      linalg.yield %14 : f16
    }
    return %arg5 : memref<?x?xf16, strided<[?, ?], offset: ?>>
  }
}

