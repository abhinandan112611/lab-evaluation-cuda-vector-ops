import subprocess, csv, os

os.makedirs("results", exist_ok=True)
header = ["op","N","threads","cpu_ms","gpu_kernel_ms","gpu_total_ms",
          "speedup_kernel","speedup_total","bandwidth_GBps","errors"]

with open("results/results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for op in (0, 1):
        for n in (10_000, 100_000, 1_000_000, 10_000_000, 50_000_000):
            for t in (64, 128, 256, 512, 1024):
                for rep in range(5):
                    out = subprocess.run(["vec_gpu.exe", str(n), str(t), str(op)],
                                         capture_output=True, text=True).stdout.strip()
                    if not out:
                        continue
                    w.writerow([kv.split("=")[1] for kv in out.split(",")])
                print(f"op={op} N={n} threads={t} done")
print("ALL DONE")