import csv


def read_data(fn):
    methods = []
    gpu = []
    total = []
    util = []
    per = []
    with open(fn) as f:
        r = csv.DictReader(f)
        for row in r:
            method = row['method']
            if method in {'baseline', 'sequential'}:
                continue
            methods.append(method)
            gpu.append(float(row['gpu_kernel_ms']))
            total.append(int(row['total_allocated_sm']))
            util.append(float(row.get('sm_utilization_pct', 100.0 * int(row['total_allocated_sm']) / 114.0)))
            per.append([int(row['sm_stream0']), int(row['sm_stream1']), int(row['sm_stream2']), int(row['sm_stream3']), int(row['sm_stream4'])])
    return methods, gpu, total, util, per


def make_bar_svg(filename, title, ylabel, labels, values, unit, precision=2, speedups=None):
    w = 800
    h = 400
    pad = 70
    n = len(values)
    bw = (w - 2 * pad) / (n * 1.2)
    maxv = max(values) * 1.1 if values else 1.0

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    svg.append(f'<rect width="100%" height="100%" fill="#fff"/>')
    svg.append(f'<text x="{w / 2}" y="30" text-anchor="middle" font-size="16">{title}</text>')
    for i, v in enumerate(values):
        x = pad + i * (bw * 1.2)
        y = pad + (h - 2 * pad) * (1 - v / maxv) if maxv > 0 else pad
        height = h - 2 * pad - (y - pad) if maxv > 0 else 0
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{height:.1f}" fill="#4c78a8" stroke="#333"/>')
        svg.append(f'<text x="{x + bw / 2:.1f}" y="{h - 10}" text-anchor="middle" font-size="12">{labels[i]}</text>')
        label = f'{v:.{precision}f}{unit}'
        if speedups is not None:
            label += f'\n{speedups[i]:.2f}x'
        svg.append(f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="12">{label}</text>')
    svg.append('</svg>')
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))


def make_stacked_svg(filename, title, labels, per_stream):
    w = 800
    h = 400
    pad = 90
    n = len(labels)
    bw = (w - 2 * pad) / n
    maxv = max(sum(s) for s in per_stream) if per_stream else 1
    colors = ['#ff9999', '#99ff99', '#9999ff', '#ffcc99', '#c299ff']
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    svg.append(f'<rect width="100%" height="100%" fill="#fff"/>')
    svg.append(f'<text x="{w / 2}" y="30" text-anchor="middle" font-size="16">{title}</text>')
    for i, vals in enumerate(per_stream):
        x = pad + i * bw
        bottom = h - pad
        cum = 0
        for j, v in enumerate(vals):
            if maxv <= 0:
                hgt = 0
                y = bottom
            else:
                hgt = (v / maxv) * (h - 2 * pad)
                y = bottom - (cum + v) / maxv * (h - 2 * pad)
            svg.append(f'<rect x="{x + 5:.1f}" y="{y:.1f}" width="{bw - 10:.1f}" height="{hgt:.1f}" fill="{colors[j % len(colors)]}" stroke="#333"/>')
            if v > 0:
                svg.append(f'<text x="{x + bw / 2:.1f}" y="{y + hgt / 2 + 5:.1f}" text-anchor="middle" font-size="10">{v}</text>')
            cum += v
        svg.append(f'<text x="{x + bw / 2:.1f}" y="{h - 20}" text-anchor="middle" font-size="12">{labels[i]}</text>')

    for j, c in enumerate(colors):
        lx = w - pad + 10
        ly = pad + j * 20
        svg.append(f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" fill="{c}" stroke="#333"/>')
        svg.append(f'<text x="{lx + 20}" y="{ly}" font-size="12">stream{j}</text>')
    svg.append('</svg>')
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))


methods, gpu_ms, total_sm, sm_util_pct, per_stream = read_data('compare_data.csv')
sequential_ms = None
with open('compare_data.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        if row['method'] in {'baseline', 'sequential'}:
            sequential_ms = float(row['gpu_kernel_ms'])
            break
if sequential_ms is None:
    sequential_ms = max(gpu_ms) if gpu_ms else 1.0
relative_time = [value / sequential_ms for value in gpu_ms]

make_bar_svg('compare_gpu_time.svg', 'Relative execution time', 'x', methods, relative_time, 'x', precision=2)
make_bar_svg('compare_total_sm.svg', 'SM utilization by method', '%', methods, sm_util_pct, '%', precision=1)
make_stacked_svg('compare_per_stream_sm.svg', 'Per-stream SM allocation', methods, per_stream)
print('SVG files generated: compare_gpu_time.svg, compare_total_sm.svg, compare_per_stream_sm.svg')
