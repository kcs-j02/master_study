import csv
import math

def read_data(fn):
    methods=[]; gpu=[]; total=[]; per=[]
    with open(fn) as f:
        r=csv.DictReader(f)
        for row in r:
            methods.append(row['method'])
            gpu.append(float(row['gpu_kernel_ms']))
            total.append(int(row['total_allocated_sm']))
            per.append([int(row['sm_stream0']),int(row['sm_stream1']),int(row['sm_stream2']),int(row['sm_stream3']),int(row['sm_stream4'])])
    return methods,gpu,total,per

methods,gpu_ms,total_sm,per_stream = read_data('compare_data.csv')
baseline=gpu_ms[0]
speedups=[baseline/x for x in gpu_ms]

# simple bar svg
def make_bar_svg(filename,title,ylabel,labels,values,annot):
    w=800; h=400; pad=60
    n=len(values); bw=(w-2*pad)/(n*1.2)
    maxv=max(values)*1.1
    def tx(x): return pad + x
    def ty(y): return pad + (h-2*pad)*(1 - y/maxv)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    svg.append(f'<rect width="100%" height="100%" fill="#fff"/>')
    svg.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="16">{title}</text>')
    # bars
    for i,v in enumerate(values):
        x=pad + i*(bw*1.2)
        y=ty(v)
        height=h-pad - y
        svg.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{height}" fill="#66a3ff" stroke="#333"/>')
        svg.append(f'<text x="{x+bw/2}" y="{h-10}" text-anchor="middle" font-size="12">{labels[i]}</text>')
        svg.append(f'<text x="{x+bw/2}" y="{y-5}" text-anchor="middle" font-size="12">{annot[i]}</text>')
    svg.append('</svg>')
    with open(filename,'w') as f: f.write('\n'.join(svg))

# 1: GPU kernel time (ms) with speedup annotation
annot=[f"{gpu_ms[i]:.1f} ms\n{speedups[i]:.2f}x" for i in range(len(gpu_ms))]
make_bar_svg('compare_gpu_time.svg','GPU kernel time by method','ms',methods,gpu_ms,annot)
# 2: total SM
annot2=[str(x) for x in total_sm]
make_bar_svg('compare_total_sm.svg','Total allocated SM by method','SM',methods,total_sm,annot2)
# 3: per-stream stacked: create stacked svg

def make_stacked_svg(filename,title,labels,per_stream):
    w=800; h=400; pad=80
    n=len(labels)
    bw=(w-2*pad)/n
    maxv=max(sum(s) for s in per_stream)
    colors=['#ff9999','#99ff99','#9999ff','#ffcc99','#c299ff']
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    svg.append(f'<rect width="100%" height="100%" fill="#fff"/>')
    svg.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="16">{title}</text>')
    for i,vals in enumerate(per_stream):
        x=pad + i*bw
        bottom=h-pad
        cum=0
        for j,v in enumerate(vals):
            hgt = (v/maxv)*(h-2*pad) if maxv>0 else 0
            y=bottom - (cum + v)/maxv*(h-2*pad) if maxv>0 else bottom
            # compute y properly
            y = bottom - (cum+v)/maxv*(h-2*pad) if maxv>0 else bottom
            height = (v/maxv)*(h-2*pad) if maxv>0 else 0
            svg.append(f'<rect x="{x}" y="{y}" width="{bw-10}" height="{height}" fill="{colors[j%len(colors)]}" stroke="#333"/>')
            if v>0:
                svg.append(f'<text x="{x+ (bw-10)/2}" y="{y+height/2+5}" text-anchor="middle" font-size="10">{v}</text>')
            cum += v
        svg.append(f'<text x="{x + (bw-10)/2}" y="{h-20}" text-anchor="middle" font-size="12">{labels[i]}</text>')
    # legend
    for j,c in enumerate(colors):
        lx = w - pad + 10
        ly = pad + j*20
        svg.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" fill="{c}" stroke="#333"/>')
        svg.append(f'<text x="{lx+20}" y="{ly}" font-size="12">stream{j}</text>')
    svg.append('</svg>')
    with open(filename,'w') as f: f.write('\n'.join(svg))

make_stacked_svg('compare_per_stream_sm.svg','Per-stream SM allocation',methods,per_stream)
print('SVG files generated: compare_gpu_time.svg, compare_total_sm.svg, compare_per_stream_sm.svg')
