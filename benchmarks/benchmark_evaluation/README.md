# 4手法の共通ベンチマーク評価

`benchmarks/KESCO/`と`benchmarks/STG/`以下に配置した11個の実行可能なベンチマークを、次の4手法で評価します。

- `sequential_method`
- `existing_method`
- `existing_method_green_context`
- `proposed_method`

評価コードはこのフォルダに置き、4手法のアルゴリズムフォルダ内は変更しません。

## 全ベンチマークを一括評価

```bash
cd /home/kobayashi/main/master_study/benchmarks
./run_all_benchmarks.sh
```

既定では各組合せを12回実行し、先頭2回をウォームアップとして集計から除外します。

```bash
RUNS=20 WARMUP_RUNS=5 ./run_all_benchmarks.sh
```

短時間の動作確認は次のように実行できます。

```bash
RUNS=2 WARMUP_RUNS=1 ./run_all_benchmarks.sh
```

最新の結果は、合成STGとその他のベンチマークに分け、毎回同じ固定パスへ上書き出力します。

```text
../../results/method_comparison_figures/
├── RESULT_stg/
│   ├── all_stg_results.csv
│   └── all_stg_execution_time_comparison.png
└── result_BENCHMARK/
    ├── all_benchmark_results.csv
    └── all_benchmark_execution_time_comparison.png
```

`RESULT_stg/`には`benchmarks/STG/`以下の5件、`result_BENCHMARK/`にはそれ以外の実行可能な6件をまとめます。全11件を混在させたグラフは生成しません。

ログを含む最新の一括測定結果は`../../results/benchmark_evaluation/latest-all/`へ保存します。次回実行時は同じフォルダを更新するため、実行回数に応じて履歴フォルダが増えることはありません。

```text
latest-all/
├── sequential_method/                   # 手法別CSV・ログ
├── existing_method/
├── existing_method_green_context/
├── proposed_method/
└── all_results.csv                      # 11ベンチマーク×4手法の統合履歴
```

## 1つのベンチマークを4手法で評価

各ベンチマークフォルダの`run_evaluation.sh`を実行します。

```bash
cd /home/kobayashi/main/master_study/benchmarks/KESCO/BENCHMARK_vector_square
./run_evaluation.sh
```

最新のCSVとグラフは、ベンチマークのグループに応じて`RESULT_stg/`または`result_BENCHMARK/`へ出力します。手法別CSV・ログは`../../results/benchmark_evaluation/latest-<ベンチマーク名>/`へ保存し、次回実行時に更新します。

別のベンチマークも同じ構成です。

```bash
KESCO/ または STG/
└── BENCHMARK_<ベンチマーク名>/
    ├── <ベンチマーク名>.stg
    └── run_evaluation.sh
```

実行回数は一括評価と同じ環境変数で変更できます。

```bash
cd /home/kobayashi/main/master_study/benchmarks/KESCO/BENCHMARK_hits
RUNS=5 WARMUP_RUNS=1 ./run_evaluation.sh
```

## 評価対象ベンチマーク

KESCO由来の入力は`../KESCO/BENCHMARK_<ベンチマーク名>/`、合成STG入力は`../STG/BENCHMARK_<ベンチマーク名>/`にあります。

| ベンチマーク名 | 入力ファイル |
|---|---|
| `vector_square` | `../KESCO/BENCHMARK_vector_square/vector_square.stg` |
| `black_scholes` | `../KESCO/BENCHMARK_black_scholes/black_scholes.stg` |
| `machine_learning` | `../KESCO/BENCHMARK_machine_learning/machine_learning.stg` |
| `hits` | `../KESCO/BENCHMARK_hits/hits.stg` |
| `image_processing` | `../KESCO/BENCHMARK_image_processing/image_processing.stg` |
| `deep_learning` | `../KESCO/BENCHMARK_deep_learning/deep_learning.stg` |
| `synthetic_fully_parallel` | `../STG/BENCHMARK_synthetic_fully_parallel/synthetic_fully_parallel.stg` |
| `synthetic_mixed_chain_parallel` | `../STG/BENCHMARK_synthetic_mixed_chain_parallel/synthetic_mixed_chain_parallel.stg` |
| `synthetic_multiple_long_branches` | `../STG/BENCHMARK_synthetic_multiple_long_branches/synthetic_multiple_long_branches.stg` |
| `synthetic_random_dag` | `../STG/BENCHMARK_synthetic_random_dag/synthetic_random_dag.stg` |
| `synthetic_sequential_chain` | `../STG/BENCHMARK_synthetic_sequential_chain/synthetic_sequential_chain.stg` |

Micro-1/Micro-2は公開資料に元のDAGがないため、`../KESCO/BENCHMARK_micro_1/unavailable.md`と`../KESCO/BENCHMARK_micro_2/unavailable.md`に理由だけを記録しており、実行対象には含めません。

## 1手法だけを全ベンチマークで評価

```bash
./scripts/run_sequential_method.sh
./scripts/run_existing_method_only.sh
./scripts/run_existing_method_green_context.sh
./scripts/run_proposed_method.sh
```

この場合も`../../results/benchmark_evaluation/latest-<手法名>/`を更新します。保存先を明示的に残したい場合だけ、`RESULTS_DIR=/任意のパス`を指定してください。

## 評価値

各実行ファイルが出力する`gpu_submit_wait_ms`を収集し、ウォームアップを除いた平均・最小・最大をCSVへ記録します。グラフの棒は平均、エラーバーは最小から最大の範囲です。

現在の4手法が実行するのは各実装内のlight/heavy合成CUDAカーネルです。したがって、元アプリケーションのエンドツーエンド性能ではなく、同じタスクグラフを入力したスケジューリング手法の比較です。
