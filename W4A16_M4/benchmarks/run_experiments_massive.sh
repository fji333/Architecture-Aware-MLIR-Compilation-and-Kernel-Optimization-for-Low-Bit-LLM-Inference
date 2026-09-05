#!/bin/bash
echo "================================================================================"
echo ">>> W4A16 M4 CPU 超大规模对比实验库 (支持 L1+L2 级联分块) <<<"
echo "测试矩阵大小: M=1024, N=16384 (W矩阵=32MB), K=4096"
echo "编译参数: -O3 -mcpu=native -std=c++11"
echo "================================================================================"
printf "%-26s | %-18s | %-4s | %-10s | %-10s\n" "实验策略" "分块(L2/L1)" "Uk" "耗时(ms)" "算力(GFLOPS)"
echo "--------------------------------------------------------------------------------"

run_exp() {
    local L2_MC=$1
    local L2_NC=$2
    local L1_MC=$3
    local L1_NC=$4
    local UK=$5
    local desc=$6
    
    clang++ -O3 -mcpu=native -std=c++11 benchmark_massive.cpp -DL2_MC=$L2_MC -DL2_NC=$L2_NC -DL1_MC=$L1_MC -DL1_NC=$L1_NC -DUK=$UK -o exp_massive_bin
    
    output=$(./exp_massive_bin)
    
    time_ms=$(echo "$output" | grep "耗时" | awk '{print $2}')
    gflops=$(echo "$output" | grep "算力" | awk '{print $5}')
    
    # 格式化输出分块大小
    local tile_str=""
    if [ "$L2_MC" -eq 0 ] && [ "$L1_MC" -eq 0 ]; then
        tile_str="0x0 (无)"
    elif [ "$L2_MC" -eq 0 ]; then
        tile_str="L1:${L1_MC}x${L1_NC}"
    elif [ "$L1_MC" -eq 0 ]; then
        tile_str="L2:${L2_MC}x${L2_NC}"
    else
        tile_str="${L2_MC}x${L2_NC}/${L1_MC}x${L1_NC}"
    fi

    printf "%-26s | %-18s | %-4s | %-10s | %-10s\n" "$desc" "$tile_str" "$UK" "${time_ms}" "${gflops}"
}

echo "【第 1 组: 无分块 (裸奔 / 全局线性扫描)】"
run_exp 0 0 0 0 2 "裸奔 (基准)" 2
run_exp 0 0 0 0 4 "裸奔 (高ILP)" 4
run_exp 0 0 0 0 6 "裸奔 (极限 ILP巅峰)" 6
echo "--------------------------------------------------------------------------------"

echo "【第 2 组: 单层分块 (只用 L2 或 L1, 固定 Uk=2)】"
run_exp 0 0 64 64 2 "仅 L1 级 (64x64)" 2
run_exp 128 256 0 0 2 "仅 L2 级 (128x256)" 2
run_exp 256 512 0 0 2 "仅 L2 级 (256x512)" 2
echo "--------------------------------------------------------------------------------"

echo "【第 3 组: 双层级联分块 (严密对比：外层 L2 + 内层 L1, 固定 Uk=2)】"
run_exp 128 256 32 64 2 "L2黄金点 + L1保守" 2
run_exp 128 256 64 64 2 "L2黄金点 + L1标准" 2
run_exp 256 512 64 128 2 "L2极限 + L1极限" 2
echo "--------------------------------------------------------------------------------"

echo "【第 4 组: 双层级联分块下的极限 ILP 测试 (固定 L2=128x256, L1=64x64)】"
run_exp 128 256 64 64 4 "双层分块 Uk=4" 4
run_exp 128 256 64 64 6 "双层分块 Uk=6" 6
run_exp 128 256 64 64 8 "双层分块 Uk=8 (溢出)" 8
echo "--------------------------------------------------------------------------------"

rm exp_massive_bin
echo "全部多层级联分块实验完成！"
