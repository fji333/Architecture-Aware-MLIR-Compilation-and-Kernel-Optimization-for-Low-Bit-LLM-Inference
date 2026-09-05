import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
plt.figure(figsize=(12, 8), facecolor='#1e1e2e')
ax = plt.gca()
ax.set_facecolor('#1e1e2e')
ax.tick_params(colors='#cdd6f4', which='both')
for spine in ax.spines.values():
    spine.set_color('#45475a')

# Constants
bandwidth = 120.0  # GB/s
peak_flops = 281.6  # GFLOPS
w4a16_flops = 155.04
fp16_flops = 220.85

ai = np.logspace(-1, 2, 500)
perf_bound = np.minimum(bandwidth * ai, peak_flops)

# Plot Roofline
plt.plot(ai, perf_bound, color='#f38ba8', linewidth=3, label='Theoretical Roofline (FP16)')
plt.plot(ai, bandwidth * ai, color='#f9e2af', linewidth=2, linestyle='--', alpha=0.5, label='Memory Bandwidth Bound (120 GB/s)')
plt.axhline(y=peak_flops, color='#f38ba8', linewidth=2, linestyle='--', alpha=0.5, label='Compute Bound (281.6 GFLOPS)')

# Data points
ai_w4a16 = 10.66
ai_fp16 = 5.33

plt.scatter(ai_w4a16, w4a16_flops, color='#89b4fa', s=150, zorder=5, edgecolors='white', linewidth=2)
plt.annotate(f'W4A16 Kernel\n{w4a16_flops} GFLOPS\n(Port Contention)', xy=(ai_w4a16, w4a16_flops), 
             xytext=(ai_w4a16 + 2, w4a16_flops - 30), color='#89b4fa', fontsize=12, fontweight='bold',
             arrowprops=dict(facecolor='#89b4fa', edgecolor='#89b4fa', arrowstyle='->', linewidth=2))

plt.scatter(ai_fp16, fp16_flops, color='#a6e3a1', s=150, zorder=5, edgecolors='white', linewidth=2)
plt.annotate(f'Pure FP16 Control\n{fp16_flops} GFLOPS\n(Dequant Removed)', xy=(ai_fp16, fp16_flops), 
             xytext=(ai_fp16 - 3.5, fp16_flops - 40), color='#a6e3a1', fontsize=12, fontweight='bold',
             arrowprops=dict(facecolor='#a6e3a1', edgecolor='#a6e3a1', arrowstyle='->', linewidth=2))

# Ridge point
ridge_point = peak_flops / bandwidth
plt.scatter(ridge_point, peak_flops, color='#cba6f7', s=100, zorder=5)
plt.annotate(f'Ridge Point\n{ridge_point:.2f} FLOPs/B', xy=(ridge_point, peak_flops), 
             xytext=(ridge_point - 1.5, peak_flops + 20), color='#cba6f7', fontsize=11)

# Highlight zones
plt.axhspan(w4a16_flops, fp16_flops, color='#89b4fa', alpha=0.1, label='ALU Port Contention Penalty (~65 GFLOPS)')
plt.axhspan(fp16_flops, peak_flops, color='#a6e3a1', alpha=0.1, label='TLB/Fabric/DVFS Penalty (~61 GFLOPS)')

# Formatting
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.ylim(10, 500)
plt.xlim(0.5, 64)
plt.grid(True, which="both", ls="--", color='#45475a', alpha=0.5)

plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', color='#cdd6f4', fontsize=14, fontweight='bold')
plt.ylabel('Performance (GFLOPS)', color='#cdd6f4', fontsize=14, fontweight='bold')
plt.title('Apple M4 Single-Core P-Core Roofline Model\nW4A16 vs Pure FP16 GEMM Analysis', color='#cdd6f4', fontsize=16, fontweight='bold', pad=20)

plt.legend(loc='lower right', facecolor='#11111b', edgecolor='#45475a', labelcolor='#cdd6f4', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/a15583507331/Desktop/M4_Roofline_Model.png', dpi=300, bbox_inches='tight', facecolor=ax.get_facecolor())
plt.close()
