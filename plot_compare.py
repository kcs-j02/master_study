import csv
import matplotlib.pyplot as plt
import numpy as np

data_file = 'compare_data.csv'
methods = []
gpu_ms = []
total_sm = []
per_stream = []
with open(data_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        methods.append(row['method'])
        gpu_ms.append(float(row['gpu_kernel_ms']))
        total_sm.append(float(row['total_allocated_sm']))
        per_stream.append([int(row['sm_stream0']), int(row['sm_stream1']), int(row['sm_stream2']), int(row['sm_stream3']), int(row['sm_stream4'])])

baseline = gpu_ms[0]
speedups = [baseline / x for x in gpu_ms]

# Plot 1: gpu kernel ms
x = np.arange(len(methods))
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(x, gpu_ms, color=['#777777','#2ca02c','#1f77b4','#ff7f0e'])
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel('GPU kernel time (ms)')
ax.set_title('GPU kernel time by method')
for i, b in enumerate(bars):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f"{gpu_ms[i]:.1f} ms\n{speedups[i]:.2f}x", ha='center', va='bottom')
fig.tight_layout()
plt.savefig('compare_gpu_time.png', dpi=150)

# Plot 2: total allocated SM
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(x, total_sm, color=['#777777','#2ca02c','#1f77b4','#ff7f0e'])
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel('Total allocated SM')
ax.set_title('Total allocated SM by method')
for i, b in enumerate(bars):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{total_sm[i]:.0f}", ha='center')
fig.tight_layout()
plt.savefig('compare_total_sm.png', dpi=150)

# Plot 3: per-stream stacked (for methods with per-stream data)
streams = ['stream0','stream1','stream2','stream3','stream4']
per_stream_arr = np.array(per_stream)
fig, ax = plt.subplots(figsize=(8,4))
bottom = np.zeros(len(methods))
colors = plt.cm.Pastel1(np.linspace(0,1,5))
for si in range(5):
    vals = per_stream_arr[:,si]
    ax.bar(x, vals, bottom=bottom, color=colors[si], label=streams[si])
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel('SM count per stream')
ax.set_title('Per-stream SM allocation')
ax.legend()
fig.tight_layout()
plt.savefig('compare_per_stream_sm.png', dpi=150)

print('Saved compare_gpu_time.png, compare_total_sm.png, compare_per_stream_sm.png')
