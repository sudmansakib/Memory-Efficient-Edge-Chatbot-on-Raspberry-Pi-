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

# Raspberry Pi interactive tokens/sec results
cache_types = ["None", "Sliding", "Paged", "Quantized"]
tokens_sec = [8.77, 8.80, 8.85, 8.71]

plt.figure(figsize=(6, 4))
bars = plt.bar(cache_types, tokens_sec, color=["#4C72B0"])

plt.title("Raspberry Pi: Tokens/sec vs Cache Type", fontsize=12)
plt.xlabel("Cache Type", fontsize=11)
plt.ylabel("Tokens/sec", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate bars with numeric values
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("tokens_sec_vs_cache_pi.png", dpi=300)
plt.show()

# Raspberry Pi interactive RSS results
cache_types = ["None", "Sliding", "Paged", "Quantized"]
rss_mb = [685.1, 684.8, 685.1, 686.1]

plt.figure(figsize=(6, 4))
bars = plt.bar(cache_types, rss_mb, color=["#DD8452"])

plt.title("Raspberry Pi: Peak RSS vs Cache Type", fontsize=12)
plt.xlabel("Cache Type", fontsize=11)
plt.ylabel("Peak RSS (MB)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate bars with numeric values
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.2,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("peak_rss_vs_cache_pi.png", dpi=300)
plt.show()

# Raspberry Pi interactive results
cache_types = ["None", "Sliding", "Paged", "Quantized"]

rss_mb = [685.1, 684.8, 685.1, 686.1]
tokens_sec = [8.77, 8.80, 8.85, 8.71]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Subplot 1: Peak RSS ---
ax = axes[0]
bars = ax.bar(cache_types, rss_mb, color="#DD8452")
ax.set_title("Peak RSS vs Cache Type (Pi)", fontsize=12)
ax.set_ylabel("Peak RSS (MB)", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate values
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.2,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# --- Subplot 2: Tokens/sec ---
ax = axes[1]
bars = ax.bar(cache_types, tokens_sec, color="#4C72B0")
ax.set_title("Tokens/sec vs Cache Type (Pi)", fontsize=12)
ax.set_ylabel("Tokens/sec", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate values
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.05,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("pi_memory_and_speed.png", dpi=300)
plt.show()

cache_types = ["None", "Sliding", "Paged", "Quantized"]
rss_mb = [685.38, 701.83, 701.72, 691.58]

plt.figure(figsize=(6,4))
bars = plt.bar(cache_types, rss_mb, color="#DD8452")

plt.title("Raspberry Pi (Fixed Prompt): Peak RSS vs Cache Type", fontsize=12)
plt.xlabel("Cache Type")
plt.ylabel("Peak RSS (MB)")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 0.5,
             f"{height:.2f}",
             ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("peak_rss_vs_cache_pi_fixed.png", dpi=300)
plt.show()

cache_types = ["None", "Sliding", "Paged", "Quantized"]
tokens_sec = [8.74, 8.51, 8.70, 8.48]

plt.figure(figsize=(6,4))
bars = plt.bar(cache_types, tokens_sec, color="#4C72B0")

plt.title("Raspberry Pi (Fixed Prompt): Tokens/sec vs Cache Type", fontsize=12)
plt.xlabel("Cache Type")
plt.ylabel("Tokens/sec")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 0.05,
             f"{height:.2f}",
             ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("tokens_sec_vs_cache_pi_fixed.png", dpi=300)
plt.show()