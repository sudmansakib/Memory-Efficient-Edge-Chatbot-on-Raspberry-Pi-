import matplotlib.pyplot as plt

cache_types = ["None", "Sliding", "Paged", "Quantized"]
tokens_sec = [42.99, 45.86, 43.30, 39.98]

plt.figure(figsize=(6,4))
plt.bar(cache_types, tokens_sec)
plt.title("Tokens/sec vs Cache Type (Baseline CPU)")
plt.xlabel("Cache Type")
plt.ylabel("Tokens/sec")
plt.tight_layout()
plt.show()

cache_types = ["None", "Sliding", "Paged", "Quantized"]
rss = [648.05, 693.80, 693.84, 653.15]

plt.figure(figsize=(6,4))
plt.bar(cache_types, rss)
plt.title("Peak RSS (MB) vs Cache Type (Baseline CPU)")
plt.xlabel("Cache Type")
plt.ylabel("Peak RSS (MB)")
plt.tight_layout()
plt.show()

# Data from CPU interactive benchmark
cache_types = ["None", "Sliding", "Paged", "Quantized"]
tokens_sec = [108.49, 132.83, 117.38, 38.88]

plt.figure(figsize=(6, 4))
bars = plt.bar(cache_types, tokens_sec)

plt.title("Tokens/sec vs Cache Type (CPU Interactive)", fontsize=12)
plt.xlabel("Cache Type")
plt.ylabel("Tokens/sec")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("tokens_sec_vs_cache_cpu_interactive.png", dpi=300)
plt.show()

# Data from CPU interactive benchmark
cache_types = ["None", "Sliding", "Paged", "Quantized"]
rss_values = [1577.8, 1578.4, 1576.9, 1289.0]

plt.figure(figsize=(6, 4))
bars = plt.bar(cache_types, rss_values)

plt.title("Peak RSS vs Cache Type (CPU Interactive)", fontsize=12)
plt.xlabel("Cache Type")
plt.ylabel("Peak RSS (MB)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("peak_rss_vs_cache_cpu_interactive.png", dpi=300)
plt.show()