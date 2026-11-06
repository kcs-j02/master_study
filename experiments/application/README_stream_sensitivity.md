# Image processing / Deep Learning stream sensitivity

This experiment fixes the strict-ablation `full` pipeline to each Stream count
from 1 through 5. It isolates whether the two applications respond similarly
to the number of Streams.

```bash
RUNS=12 WARMUP_RUNS=2 \
./experiments/application/stream_sensitivity.sh
python3 experiments/application/analyze_image_deep.py
```

The output is `results/application_stream_sensitivity/stream_sensitivity.csv`
and `paper/evidence_figures/image_deep_stream_sensitivity.png`.
