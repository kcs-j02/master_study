# Image processing and Deep Learning analysis

## Graph structure

| Metric | Image processing | Deep Learning |
|---|---:|---:|
| Tasks | 11 | 8 |
| Edges | 11 | 7 |
| Roots | 3 | 2 |
| Leaves | 1 | 1 |
| Levels | 6 | 5 |
| Maximum level width | 3 | 2 |
| Total work | 254 | 786 |
| Critical path | 202 | 698 |
| Work / critical path | 1.26 | 1.13 |

## Interpretation

Image processing has more tasks and a wider branching structure. The sensitivity experiment shows a clear improvement through four Streams, consistent with useful independent image-processing branches. Deep Learning has two long convolution branches that merge at concat and dot-product tasks. Its best fixed Stream count is also four, but the improvement is smaller and variable; the existing Green Context method remains faster in the four-method comparison. Therefore, the current evidence supports a conditional claim: the proposed allocation is useful when branch-level parallelism and task durations are balanced enough for the selected SM partition to reduce the critical-path bottleneck, but it does not dominate for the Deep Learning graph.
