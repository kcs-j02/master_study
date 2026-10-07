import csv
from pathlib import Path


study_root = Path(__file__).resolve().parents[2]
output_dir = study_root / 'results' / 'archived_comparison'
output_dir.mkdir(parents=True, exist_ok=True)


def pretty_method(method):
    names = {
        'baseline': 'SEQ',
        'sequential': 'SEQ',
        'ks': 'KS',
        'ks_gc': 'KS+GC',
        'ks+gc': 'KS+GC',
        'proposed': 'PR',
        'pr': 'PR',
    }
    return names.get(method.lower(), method)


def read_execution_data(fn):
    methods = []
    gpu_ms = []

    with open(fn, encoding='utf-8') as f:
        r = csv.DictReader(f)

        for row in r:
            methods.append(pretty_method(row['method']))
            gpu_ms.append(float(row['gpu_kernel_ms']))

    return methods, gpu_ms


def read_sm_data(fn):
    methods = []
    util = []
    per = []

    with open(fn, encoding='utf-8') as f:
        r = csv.DictReader(f)

        for row in r:
            method = row['method']

            # SM比較ではSEQを除外
            if method.lower() in {'baseline', 'sequential'}:
                continue

            methods.append(pretty_method(method))

            total_sm = int(row['total_allocated_sm'])

            util.append(
                float(
                    row.get(
                        'sm_utilization_pct',
                        100.0 * total_sm / 114.0
                    )
                )
            )

            per.append([
                int(row['sm_stream0']),
                int(row['sm_stream1']),
                int(row['sm_stream2']),
                int(row['sm_stream3']),
                int(row['sm_stream4']),
            ])

    return methods, util, per


def make_speedup_time_svg(
    filename,
    methods,
    gpu_ms,
):
    """
    左 : Speedup
    右 : Execution Time
    """

    w = 1200
    h = 450

    top = 70
    bottom = 70

    panel_width = 500
    gap = 100

    left_x = 70
    right_x = left_x + panel_width + gap

    plot_h = h - top - bottom

    # SEQを基準にする
    sequential_ms = None

    for method, value in zip(methods, gpu_ms):
        if method == 'SEQ':
            sequential_ms = value
            break

    if sequential_ms is None:
        raise ValueError(
            'SEQ (baseline/sequential) が compare_data.csv にありません'
        )

    speedups = [
        sequential_ms / value
        for value in gpu_ms
    ]

    max_speedup = max(speedups) * 1.15
    max_time = max(gpu_ms) * 1.15

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}">'
    ]

    svg.append(
        '<rect width="100%" height="100%" fill="#fff"/>'
    )

    # =====================================================
    # タイトル
    # =====================================================

    svg.append(
        f'<text x="{left_x + panel_width / 2}" '
        f'y="35" text-anchor="middle" '
        f'font-size="22" font-weight="bold">'
        f'Speedup'
        f'</text>'
    )

    svg.append(
        f'<text x="{right_x + panel_width / 2}" '
        f'y="35" text-anchor="middle" '
        f'font-size="22" font-weight="bold">'
        f'Execution Time'
        f'</text>'
    )

    # =====================================================
    # 左：Speedup
    # =====================================================

    n = len(methods)
    bar_space = panel_width / n
    bar_width = bar_space * 0.65

    baseline_y = top + plot_h

    # 軸
    svg.append(
        f'<line x1="{left_x}" y1="{top}" '
        f'x2="{left_x}" y2="{baseline_y}" '
        f'stroke="#333"/>'
    )

    svg.append(
        f'<line x1="{left_x}" y1="{baseline_y}" '
        f'x2="{left_x + panel_width}" y2="{baseline_y}" '
        f'stroke="#333"/>'
    )

    # 1.0x 基準線
    one_y = baseline_y - (1.0 / max_speedup) * plot_h

    svg.append(
        f'<line x1="{left_x}" y1="{one_y:.1f}" '
        f'x2="{left_x + panel_width}" y2="{one_y:.1f}" '
        f'stroke="#999" stroke-dasharray="5,5"/>'
    )

    svg.append(
        f'<text x="{left_x - 8}" y="{one_y + 5:.1f}" '
        f'text-anchor="end" font-size="13">'
        f'1.0'
        f'</text>'
    )

    for i, (method, speedup) in enumerate(
        zip(methods, speedups)
    ):
        x = (
            left_x
            + i * bar_space
            + (bar_space - bar_width) / 2
        )

        bar_h = (speedup / max_speedup) * plot_h
        y = baseline_y - bar_h

        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{bar_width:.1f}" '
            f'height="{bar_h:.1f}" '
            f'fill="#4c78a8"/>'
        )

        # 値
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" '
            f'y="{y - 8:.1f}" '
            f'text-anchor="middle" '
            f'font-size="16">'
            f'{speedup:.3f}x'
            f'</text>'
        )

        # 手法名
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" '
            f'y="{baseline_y + 25}" '
            f'text-anchor="middle" '
            f'font-size="16">'
            f'{method}'
            f'</text>'
        )

    # 縦軸ラベル
    svg.append(
        f'<text '
        f'x="20" '
        f'y="{top + plot_h / 2}" '
        f'text-anchor="middle" '
        f'font-size="17" '
        f'transform="rotate(-90 20 {top + plot_h / 2})">'
        f'Speedup [x]'
        f'</text>'
    )

    # =====================================================
    # 右：Execution Time
    # =====================================================

    svg.append(
        f'<line x1="{right_x}" y1="{top}" '
        f'x2="{right_x}" y2="{baseline_y}" '
        f'stroke="#333"/>'
    )

    svg.append(
        f'<line x1="{right_x}" y1="{baseline_y}" '
        f'x2="{right_x + panel_width}" y2="{baseline_y}" '
        f'stroke="#333"/>'
    )

    for i, (method, value) in enumerate(
        zip(methods, gpu_ms)
    ):
        x = (
            right_x
            + i * bar_space
            + (bar_space - bar_width) / 2
        )

        bar_h = (value / max_time) * plot_h
        y = baseline_y - bar_h

        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{bar_width:.1f}" '
            f'height="{bar_h:.1f}" '
            f'fill="#4c78a8"/>'
        )

        # 実行時間
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" '
            f'y="{y - 8:.1f}" '
            f'text-anchor="middle" '
            f'font-size="16">'
            f'{value:.2f} ms'
            f'</text>'
        )

        # 手法名
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" '
            f'y="{baseline_y + 25}" '
            f'text-anchor="middle" '
            f'font-size="16">'
            f'{method}'
            f'</text>'
        )

    svg.append(
        f'<text '
        f'x="{right_x - 50}" '
        f'y="{top + plot_h / 2}" '
        f'text-anchor="middle" '
        f'font-size="17" '
        f'transform="rotate(-90 '
        f'{right_x - 50} {top + plot_h / 2})">'
        f'Execution Time [ms]'
        f'</text>'
    )

    svg.append('</svg>')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))


def make_bar_svg(
    filename,
    title,
    labels,
    values,
    unit,
    precision=2
):
    w = 800
    h = 400
    pad = 70

    n = len(values)

    bw = (w - 2 * pad) / (n * 1.2)

    maxv = max(values) * 1.1 if values else 1.0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}">'
    ]

    svg.append(
        '<rect width="100%" height="100%" fill="#fff"/>'
    )

    svg.append(
        f'<text x="{w / 2}" y="30" '
        f'text-anchor="middle" font-size="16">'
        f'{title}'
        f'</text>'
    )

    for i, v in enumerate(values):
        x = pad + i * (bw * 1.2)

        y = (
            pad
            + (h - 2 * pad)
            * (1 - v / maxv)
        )

        height = (
            h
            - 2 * pad
            - (y - pad)
        )

        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{bw:.1f}" '
            f'height="{height:.1f}" '
            f'fill="#4c78a8" '
            f'stroke="#333"/>'
        )

        svg.append(
            f'<text x="{x + bw / 2:.1f}" '
            f'y="{h - 10}" '
            f'text-anchor="middle" '
            f'font-size="12">'
            f'{labels[i]}'
            f'</text>'
        )

        svg.append(
            f'<text x="{x + bw / 2:.1f}" '
            f'y="{y - 5:.1f}" '
            f'text-anchor="middle" '
            f'font-size="12">'
            f'{v:.{precision}f}{unit}'
            f'</text>'
        )

    svg.append('</svg>')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))


def make_stacked_svg(
    filename,
    title,
    labels,
    per_stream
):
    w = 800
    h = 400
    pad = 90

    n = len(labels)
    bw = (w - 2 * pad) / n

    maxv = (
        max(sum(s) for s in per_stream)
        if per_stream
        else 1
    )

    colors = [
        '#ff9999',
        '#99ff99',
        '#9999ff',
        '#ffcc99',
        '#c299ff'
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}">'
    ]

    svg.append(
        '<rect width="100%" height="100%" fill="#fff"/>'
    )

    svg.append(
        f'<text x="{w / 2}" y="30" '
        f'text-anchor="middle" '
        f'font-size="16">'
        f'{title}'
        f'</text>'
    )

    for i, vals in enumerate(per_stream):
        x = pad + i * bw
        bottom = h - pad
        cum = 0

        for j, v in enumerate(vals):
            hgt = (
                (v / maxv) * (h - 2 * pad)
                if maxv > 0
                else 0
            )

            y = (
                bottom
                - (cum + v)
                / maxv
                * (h - 2 * pad)
                if maxv > 0
                else bottom
            )

            svg.append(
                f'<rect '
                f'x="{x + 5:.1f}" '
                f'y="{y:.1f}" '
                f'width="{bw - 10:.1f}" '
                f'height="{hgt:.1f}" '
                f'fill="{colors[j % len(colors)]}" '
                f'stroke="#333"/>'
            )

            if v > 0:
                svg.append(
                    f'<text '
                    f'x="{x + bw / 2:.1f}" '
                    f'y="{y + hgt / 2 + 5:.1f}" '
                    f'text-anchor="middle" '
                    f'font-size="10">'
                    f'{v}'
                    f'</text>'
                )

            cum += v

        svg.append(
            f'<text '
            f'x="{x + bw / 2:.1f}" '
            f'y="{h - 20}" '
            f'text-anchor="middle" '
            f'font-size="12">'
            f'{labels[i]}'
            f'</text>'
        )

    svg.append('</svg>')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))


# =========================================================
# Main
# =========================================================

execution_methods, gpu_ms = read_execution_data(
    output_dir / 'compare_data.csv'
)

sm_methods, sm_util_pct, per_stream = read_sm_data(
    output_dir / 'compare_data.csv'
)

# 左：Speedup
# 右：Execution Time
make_speedup_time_svg(
    output_dir / 'compare_speedup_time.svg',
    execution_methods,
    gpu_ms
)

# SM利用率
make_bar_svg(
    output_dir / 'compare_total_sm.svg',
    'SM utilization by method',
    sm_methods,
    sm_util_pct,
    '%',
    precision=1
)

# StreamごとのSM割当
make_stacked_svg(
    output_dir / 'compare_per_stream_sm.svg',
    'Per-stream SM allocation',
    sm_methods,
    per_stream
)

print(
    'SVG files generated: '
    'compare_speedup_time.svg, '
    'compare_total_sm.svg, '
    'compare_per_stream_sm.svg'
)
