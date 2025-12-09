import matplotlib.pyplot as plt

# Sequence lengths
lengths = [16, 32, 64, 128, 256]

# Pi RSS values (MB)
none_rss =      [683.61, 683.67, 683.80, 684.00, 690.61]
sliding_rss =   [684.34, 684.41, 684.53, 684.72, 691.31]
paged_rss =     [684.30, 684.38, 684.45, 684.69, 691.28]
quantized_rss = [684.63, 684.69, 686.05, 695.47, 717.38]

plt.figure(figsize=(7,5))

plt.plot(lengths, none_rss,      marker='o', label='None')
plt.plot(lengths, sliding_rss,   marker='o', label='Sliding')
plt.plot(lengths, paged_rss,     marker='o', label='Paged')
plt.plot(lengths, quantized_rss, marker='o', label='Quantized')

plt.title("Raspberry Pi: Memory Consumption vs Sequence Length", fontsize=12)
plt.xlabel("Sequence Length (tokens)", fontsize=11)
plt.ylabel("Peak RSS (MB)", fontsize=11)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("memory_vs_length_pi_all.png", dpi=300)
plt.show()
