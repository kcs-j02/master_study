import csv
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 30,
    'axes.labelsize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
})


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

        sm_util_pct.append(
            float(
                row.get(
                    'sm_utilization_pct',
                    100.0 * float(row['total_allocated_sm']) / 114.0
                )
            )
        )

        per_stream.append([
            int(row['sm_stream0']),
            int(row['sm_stream1']),
            int(row['sm_stream2']),
            int(row['sm_stream3']),
            int(row['sm_stream4'])
        ])


x = np.arange(len(methods))

colors = [
    '#4c78a8',
    '#f58518',
    '#54a24b'
]


if sequential_ms is None:
    sequential_ms = max(gpu_ms) if gpu_ms else 1.0


relative_time = [
    value / sequential_ms
    for value in gpu_ms
]


# ============================================================
# Plot 1
# Relative execution time
# ============================================================

fig, ax = plt.subplots(figsize=(13, 8))

bars = ax.bar(
    x,
    relative_time,
    color=colors[:len(methods)]
)

ax.set_xticks(x)
ax.set_xticklabels(
    methods,
    rotation=15
)

ax.set_ylabel(
    'Relative Time\n(Sequential = 1.0)',
    fontsize=30
)

ax.set_title(
    'Relative Execution Time by Method',
    fontsize=32
)

ax.tick_params(
    axis='both',
    labelsize=24
)

for b, value, raw_ms in zip(
    bars,
    relative_time,
    gpu_ms
):
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() * 1.03,
        f'{value:.2f}x\n{raw_ms:.2f} ms',
        ha='center',
        va='bottom',
        fontsize=24,
        fontweight='bold'
    )

fig.tight_layout()

plt.savefig(
    'compare_gpu_time.png',
    dpi=200,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# Plot 2
# SM utilization
# ============================================================

fig, ax = plt.subplots(figsize=(13, 8))

bars = ax.bar(
    x,
    sm_util_pct,
    color=colors[:len(methods)]
)

ax.set_xticks(x)
ax.set_xticklabels(
    methods,
    rotation=15
)

ax.set_ylabel(
    'SM Utilization (%)',
    fontsize=30
)

ax.set_title(
    'SM Utilization by Method',
    fontsize=32
)

ax.tick_params(
    axis='both',
    labelsize=24
)

for b, value in zip(
    bars,
    sm_util_pct
):
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() * 1.02,
        f'{value:.1f}%',
        ha='center',
        va='bottom',
        fontsize=25,
        fontweight='bold'
    )

ax.set_ylim(
    0,
    max(sm_util_pct) * 1.22
)

fig.tight_layout()

plt.savefig(
    'compare_total_sm.png',
    dpi=200,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# Plot 3
# Per-stream SM allocation
# ============================================================

streams = [
    'stream0',
    'stream1',
    'stream2',
    'stream3',
    'stream4'
]

per_stream_arr = np.array(per_stream)

fig, ax = plt.subplots(figsize=(14, 8))

bottom = np.zeros(len(methods))

stream_colors = plt.cm.Set2(
    np.linspace(0, 1, 5)
)

for si in range(5):

    vals = per_stream_arr[:, si]

    bars = ax.bar(
        x,
        vals,
        bottom=bottom,
        color=stream_colors[si],
        label=streams[si],
        width=0.8
    )

    for b, v in zip(
        bars,
        vals
    ):
        if v > 0:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_y() + v / 2,
                str(v),
                ha='center',
                va='center',
                fontsize=22,
                fontweight='bold'
            )

    bottom += vals


ax.set_xticks(x)
ax.set_xticklabels(
    methods,
    rotation=15
)

ax.set_ylabel(
    'SM Allocation',
    fontsize=30
)

ax.set_title(
    'Per-Stream SM Allocation',
    fontsize=32
)

ax.tick_params(
    axis='both',
    labelsize=24
)

ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.25, 1.0),
    fontsize=22
)

fig.tight_layout()

plt.savefig(
    'compare_per_stream_sm.png',
    dpi=200,
    bbox_inches='tight'
)

plt.close()


print(
    'Saved compare_gpu_time.png, '
    'compare_total_sm.png, '
    'compare_per_stream_sm.png'
)