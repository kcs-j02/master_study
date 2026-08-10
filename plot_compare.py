import csv
import matplotlib.pyplot as plt
import numpy as np


data_file = 'compare_data.csv'
methods = []
gpu_ms = []
total_sm = []
per_stream = []
sm_util_pct = []

sequential_ms = None
with open(data_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        if method in {'baseline', 'sequential'}:
            sequential_ms = float(row['gpu_kernel_ms'])
            continue
        methods.append(method)
        gpu_ms.append(float(row['gpu_kernel_ms']))
        total_sm.append(float(row['total_allocated_sm']))
        sm_util_pct.append(float(row.get('sm_utilization_pct', 100.0 * float(row['total_allocated_sm']) / 114.0)))
        per_stream.append([int(row['sm_stream0']), int(row['sm_stream1']), int(row['sm_stream2']), int(row['sm_stream3']), int(row['sm_stream4'])])

x = np.arange(len(methods))
colors = ['#4c78a8', '#f58518', '#54a24b']

if sequential_ms is None:
    sequential_ms = max(gpu_ms) if gpu_ms else 1.0
relative_time = [value / sequential_ms for value in gpu_ms]

# Plot 1: GPU kernel time
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(x, relative_time, color=colors[:len(methods)])
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel('Relative time (sequential = 1.0)')
ax.set_title('Relative execution time by method')
for b, value, raw_ms in zip(bars, relative_time, gpu_ms):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f'{value:.2f}x\n{raw_ms:.2f} ms', ha='center', va='bottom')
fig.tight_layout()
plt.savefig('compare_gpu_time.png', dpi=150)

# Plot 2: SM utilization rate
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(x, sm_util_pct, color=colors[:len(methods)])
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel('SM utilization (%)')
ax.set_title('SM utilization by method')
for b, value in zip(bars, sm_util_pct):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f'{value:.1f}%', ha='center', va='bottom')
fig.tight_layout()
plt.savefig('compare_total_sm.png', dpi=150)

# Plot 3: per-stream stacked allocation
streams = ['stream0', 'stream1', 'stream2', 'stream3', 'stream4']
per_stream_arr = np.array(per_stream)
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(methods))
stream_colors = plt.cm.Set2(np.linspace(0, 1, 5))
for si in range(5):
    vals = per_stream_arr[:, si]
    bars = ax.bar(x, vals, bottom=bottom, color=stream_colors[si], label=streams[si], width=0.8)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_y() + v / 2, str(v), ha='center', va='center', fontsize=8)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel('SM allocation')
ax.set_title('Per-stream SM allocation')
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
fig.tight_layout()
plt.savefig('compare_per_stream_sm.png', dpi=150)

print('Saved compare_gpu_time.png, compare_total_sm.png, compare_per_stream_sm.png')
