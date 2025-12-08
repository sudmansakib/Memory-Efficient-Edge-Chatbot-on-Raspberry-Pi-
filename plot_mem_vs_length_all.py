import matplotlib.pyplot as plt

# Data from your sweeps
lengths = [16, 32, 64, 128, 256]

none_rss =      [1283.33, 1284.20, 1284.81, 1287.25, 1294.93]
sliding_rss =   [1284.80, 1285.67, 1286.19, 1288.28, 1296.33]
paged_rss =     [1282.28, 1283.28, 1283.98, 1285.95, 1293.99]
quantized_rss = [1286.64, 1287.71, 1291.50, 1301.15, 1321.29]

plt.figure(figsize=(7,5))

plt.plot(lengths, none_rss,      marker='o', label='None')
plt.plot(lengths, sliding_rss,   marker='o', label='Sliding')
plt.plot(lengths, paged_rss,     marker='o', label='Paged')
plt.plot(lengths, quantized_rss, marker='o', label='Quantized')

plt.title("Memory Consumption vs Sequence Length (CPU)")
plt.xlabel("Sequence Length (tokens)")
plt.ylabel("Peak RSS (MB)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("memory_vs_length_cpu_all.png", dpi=300)
plt.show()
