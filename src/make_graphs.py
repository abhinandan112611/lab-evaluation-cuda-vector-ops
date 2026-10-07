import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEAK_BW = 192.0  # GB/s, approximate peak for GTX 1650 Ti laptop; verify in spec sheet

os.makedirs("graphs", exist_ok=True)
df = pd.read_csv("results/results.csv")

# Average the 5 repeats for every (op, N, threads) setting
avg = df.groupby(["op", "N", "threads"], as_index=False).mean(numeric_only=True)
avg["efficiency_pct"] = avg["bandwidth_GBps"] / PEAK_BW * 100
avg.to_csv("results/averaged.csv", index=False)

best_t = 256  # block size used for the size-scaling graphs

for op in ("add", "mul"):
    d = avg[(avg["op"] == op) & (avg["threads"] == best_t)].sort_values("N")

    # 1. Execution time vs N
    plt.figure(figsize=(7, 4.5))
    plt.plot(d["N"], d["cpu_ms"], "o-", label="CPU")
    plt.plot(d["N"], d["gpu_kernel_ms"], "s-", label="GPU kernel only")
    plt.plot(d["N"], d["gpu_total_ms"], "^-", label="GPU total (with copies)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Vector size N"); plt.ylabel("Time (ms)")
    plt.title(f"Execution time vs N ({op}, {best_t} threads/block)")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"graphs/time_vs_N_{op}.png", dpi=150); plt.close()

    # 2. Speedup vs N
    plt.figure(figsize=(7, 4.5))
    plt.plot(d["N"], d["speedup_kernel"], "s-", label="Kernel-only speedup")
    plt.plot(d["N"], d["speedup_total"], "^-", label="Total speedup (with copies)")
    plt.axhline(1, color="gray", linestyle="--", label="Break-even (1x)")
    plt.xscale("log")
    plt.xlabel("Vector size N"); plt.ylabel("Speedup (CPU time / GPU time)")
    plt.title(f"Speedup vs N ({op}, {best_t} threads/block)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"graphs/speedup_vs_N_{op}.png", dpi=150); plt.close()

    # 3. Efficiency (bandwidth utilisation) vs N
    plt.figure(figsize=(7, 4.5))
    plt.plot(d["N"], d["efficiency_pct"], "o-", color="green")
    plt.xscale("log")
    plt.xlabel("Vector size N"); plt.ylabel("Efficiency (% of peak bandwidth)")
    plt.title(f"Efficiency vs N ({op}, {best_t} threads/block)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"graphs/efficiency_vs_N_{op}.png", dpi=150); plt.close()

    # 4. Effect of block size on kernel time
    plt.figure(figsize=(7, 4.5))
    for n in sorted(avg["N"].unique()):
        b = avg[(avg["op"] == op) & (avg["N"] == n)].sort_values("threads")
        plt.plot(b["threads"], b["gpu_kernel_ms"], "o-", label=f"N={n:,}")
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("Threads per block"); plt.ylabel("Kernel time (ms)")
    plt.title(f"Effect of block size on kernel time ({op})")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"graphs/blocksize_{op}.png", dpi=150); plt.close()

# Summary table at the chosen block size
summary = avg[avg["threads"] == best_t][
    ["op", "N", "cpu_ms", "gpu_kernel_ms", "gpu_total_ms",
     "speedup_kernel", "speedup_total", "bandwidth_GBps", "efficiency_pct"]
].sort_values(["op", "N"]).round(3)
summary.to_csv("results/summary.csv", index=False)
print(summary.to_string(index=False))
print("\nGraphs saved in the graphs folder.")