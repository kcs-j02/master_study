# Results report

Generate the PDF, figures, and tables from the completed application-ablation
measurements:

```bash
python3 paper/results_report/generate_report.py
```

Outputs:

- `results_report.pdf`
- `figures/application_ablation_execution_time.png`
- `figures/application_ablation_relative_change.png`
- `figures/scheduling_overhead_ratio.png`
- `tables/application_ablation_table.csv`
- `tables/overhead_table.csv`

The report uses the 33-run/3-warmup KESCO application measurements. The
scale/seed experiment is not included because only its smoke test has been
executed so far.
