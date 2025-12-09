import matplotlib.pyplot as plt
import numpy as np

# Cache types (x-axis groups)
cache_types = ["None", "Sliding", "Paged", "Quantized"]
x = np.arange(len(cache_types))
width = 0.35  # width of each bar

# CPU results (fixed prompt)
cpu_tokens = [42.99, 45.86, 43.30, 39.98]
cpu_rss =    [648.05, 693.80, 693.84, 653.15]

# Pi results (fixed prompt)
pi_tokens =  [8.74, 8.51, 8.70, 8.48]
pi_rss =     [685.38, 701.83, 701.72, 691.58]

# --- Create figure with two subplots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# -----------------------------
# Subplot 1: Tokens/sec
# -----------------------------
ax = axes[0]
bars1 = ax.bar(x - width/2, cpu_tokens, width, label="CPU", color="#4C72B0")
bars2 = ax.bar(x + width/2, pi_tokens,  width, label="Pi",  color="#DD8452")

ax.set_title("Tokens/sec: CPU vs Raspberry Pi", fontsize=12)
ax.set_xlabel("Cache Type")
ax.set_ylabel("Tokens/sec")
ax.set_xticks(x)
ax.set_xticklabels(cache_types)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()

# Annotate bars
for bars in [bars1, bars2]:
    for bar in bars:
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            y + 0.5,
            f"{y:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

# -----------------------------
# Subplot 2: Peak RSS (MB)
# -----------------------------
ax = axes[1]
bars3 = ax.bar(x - width/2, cpu_rss, width, label="CPU", color="#4C72B0")
bars4 = ax.bar(x + width/2, pi_rss,  width, label="Pi",  color="#DD8452")

ax.set_title("Peak RSS (MB): CPU vs Raspberry Pi", fontsize=12)
ax.set_xlabel("Cache Type")
ax.set_ylabel("Peak RSS (MB)")
ax.set_xticks(x)
ax.set_xticklabels(cache_types)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()

# Annotate bars
for bars in [bars3, bars4]:
    for bar in bars:
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            y + 3,
            f"{y:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()
plt.savefig("cpu_vs_pi_comparison.png", dpi=300)
plt.show()
