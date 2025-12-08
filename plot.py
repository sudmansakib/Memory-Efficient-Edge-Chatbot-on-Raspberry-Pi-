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