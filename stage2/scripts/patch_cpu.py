import os

def replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)

# 1. Patch CMakeLists.txt
cmake_path = 'triton-cpu/CMakeLists.txt'
replace_in_file(cmake_path, 'LLVMNVPTXCodeGen', 'LLVMAArch64CodeGen')
replace_in_file(cmake_path, 'LLVMAMDGPUCodeGen', 'LLVMAArch64AsmParser')
replace_in_file(cmake_path, 'LLVMAMDGPUAsmParser', 'LLVMAArch64Desc\n    LLVMAArch64Info\n    LLVMAArch64Utils\n    LLVMAArch64Disassembler\n    LLVMMIRParser')

# 2. Patch llvm.cc
llvm_cc = 'triton-cpu/python/src/llvm.cc'
with open(llvm_cc, 'r') as f:
    llvm_content = f.read()

llvm_content = llvm_content.replace('LLVMInitializeNVPTXTargetInfo();', 'LLVMInitializeAArch64TargetInfo();')
llvm_content = llvm_content.replace('LLVMInitializeNVPTXTarget();', 'LLVMInitializeAArch64Target();')
llvm_content = llvm_content.replace('LLVMInitializeNVPTXTargetMC();', 'LLVMInitializeAArch64TargetMC();')
llvm_content = llvm_content.replace('LLVMInitializeNVPTXAsmPrinter();', 'LLVMInitializeAArch64AsmPrinter();')

llvm_content = llvm_content.replace('LLVMInitializeAMDGPUTargetInfo();', '//LLVMInitializeAMDGPUTargetInfo();')
llvm_content = llvm_content.replace('LLVMInitializeAMDGPUTarget();', '//LLVMInitializeAMDGPUTarget();')
llvm_content = llvm_content.replace('LLVMInitializeAMDGPUTargetMC();', '//LLVMInitializeAMDGPUTargetMC();')
llvm_content = llvm_content.replace('LLVMInitializeAMDGPUAsmPrinter();', '//LLVMInitializeAMDGPUAsmPrinter();')
llvm_content = llvm_content.replace('LLVMInitializeAMDGPUAsmParser();', '//LLVMInitializeAMDGPUAsmParser();')

# Disable PassPlugin
llvm_content = llvm_content.replace('#include "llvm/Passes/PassPlugin.h"', '//#include "llvm/Passes/PassPlugin.h"\n#include <stdexcept>')
plugin_code = '''  auto passPlugin = llvm::PassPlugin::Load(path);
  if (!passPlugin) {
    auto err = passPlugin.takeError();
    auto errMsg = llvm::toString(std::move(err));
    throw std::runtime_error("Failed to load pass plugin: " + errMsg);
  }
  passPlugin->registerPassBuilderCallbacks(pb);'''
plugin_replacement = '''  throw std::runtime_error("PassPlugin::Load is disabled in this patched build for AArch64.");'''
llvm_content = llvm_content.replace(plugin_code, plugin_replacement)

with open(llvm_cc, 'w') as f:
    f.write(llvm_content)

# 3. Patch main.cc
main_cc = 'triton-cpu/python/src/main.cc'
with open(main_cc, 'r') as f:
    main_content = f.read()
if "void init_triton_nvidia" not in main_content:
    main_content = main_content.replace('void init_triton_ir(py::module &&m);', 'void init_triton_ir(py::module &&m);\n\nvoid init_triton_nvidia(py::module &&m) {}\nvoid init_triton_amd(py::module &&m) {}')
    with open(main_cc, 'w') as f:
        f.write(main_content)

print("Patching complete.")
