# 4手法の共通ベンチマーク評価

`benchmarks/KESCO/`と`benchmarks/STG/`以下に配置した11個の実行可能なベンチマークを、次の4手法で評価します。

- `sequential_method`
- `existing_method`
- `existing_method_green_context`
- `proposed_method`

評価コードはこのフォルダに置きます。全ベンチマークの入口スクリプトは、評価開始前に4手法の`main.cu`を自動ビルドします。

## 全ベンチマークを一括評価

```bash
cd /home/kobayashi/main/master_study/benchmarks
./run_all_benchmarks.sh
```

`run_all_benchmarks.sh`は`nvcc`とTaskflowのincludeディレクトリを確認し、次の4実行ファイルをビルドしてから評価を開始します。

```text
sequential_method/main
existing_method/main
existing_method_green_context/main
proposed_method/main
```

既定のCUDAコンパイラとTaskflowのパスは、それぞれ`/usr/local/cuda/bin/nvcc`と`/home/kobayashi/taskflow`です。環境が異なる場合は次のように変更できます。

```bash
NVCC=/path/to/nvcc \
TASKFLOW_INC=/path/to/taskflow \
./run_all_benchmarks.sh
```

## 比較条件

論文用の標準比較では、4手法で次の条件を共通化しています。

- 同じSTG入力、同じ実CUDAカーネル、同じ入力サイズ、同じGPUを使用
- 同じCUDAコンパイル条件（`nvcc -O2 -std=c++20`）を使用
- 各条件を同じ回数実行し、同じ回数のウォームアップを除外
- `gpu_kernel_ms`を、最初のタスクのCUDA Event開始から最後のタスクのCUDA Event終了までのGPU makespanとして集計
- Taskflow投入、CPU待機、Context・Stream破棄時間は主指標に含めない
- 数値検証は測定区間の外で実行し、実行時間へ加えない

方式ごとのStream配置、SM配分、スケジューリング計算は比較対象のアルゴリズム差分です。提案手法だけが短時間処理を特別扱いする適応設定は、標準比較では使用しません。追加の適応最適化を評価する場合は、標準比較とは別の実験として記録してください。

既定では各組合せを33回実行し、先頭3回をウォームアップとして除外します。残り30サンプルについて平均・中央値・標準偏差・95%信頼区間を集計します。

```bash
RUNS=20 WARMUP_RUNS=5 ./run_all_benchmarks.sh
```

短時間の動作確認は次のように実行できます。

```bash
RUNS=2 WARMUP_RUNS=1 ./run_all_benchmarks.sh
```

最新の結果は、合成STGとその他のベンチマークに分け、毎回同じ固定パスへ上書き出力します。KESCO由来の実カーネルについては、計測区間外で決定的な参照値・パイプライン不変条件と比較し、全反復の検証結果もCSVへ記録します。

```text
../../results/method_comparison_figures/
├── correctness_validation.csv
├── RESULT_stg/
│   ├── all_stg_results.csv
│   └── all_stg_execution_time_comparison.png
├── result_BENCHMARK/
│   ├── all_benchmark_results.csv
│   └── all_benchmark_execution_time_comparison.png
└── sm_utilization/                       # run_sm_utilization.shの出力
    ├── sm_active_results.csv
    ├── kesco_sm_active_comparison.png
    └── stg_sm_active_comparison.png
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

### KESCO由来ベンチマークの実カーネル

KESCO由来の6件は、KESCOが比較対象として採用した公式grCUDAベンチマークスイートのCUDA C++実装を使用します。移植元は`AlbertoParravicini/grcuda`の論文用ブランチ、commit `ee0dd4cb1bade06b25ddd7dc56b4441e01172f0e`です。BSD 3-Clauseライセンスは`../common/KESCO_LICENSE.txt`に保存しています。

| 入力 | 実行カーネル |
|---|---|
| `vector_square` | vector square、reduction |
| `black_scholes` | Black–Scholes option pricing |
| `machine_learning` | ridge regression、naive Bayes、softmax、argmax |
| `hits` | sparse matrix-vector multiplication、normalization |
| `image_processing` | Gaussian blur、Sobel、min/max、unsharpen、combine |
| `deep_learning` | convolution、mean pooling、concat、dot product |

入力サイズは元スイートのV100向け最小公式サイズに固定しています。全手法の「1タスク最大64 SM」という条件を維持するため、2D/3Dカーネルも合計64 blockで起動し、各カーネル内のgrid-stride loopで全入力を処理します。独自STGおよび`synthetic_*`入力は従来どおりLIGHT/HEAVY合成カーネルへ自動的にフォールバックします。

## 1手法だけを全ベンチマークで評価

```bash
./scripts/run_sequential_method.sh
./scripts/run_existing_method_only.sh
./scripts/run_existing_method_green_context.sh
./scripts/run_proposed_method.sh
```

この場合も`../../results/benchmark_evaluation/latest-<手法名>/`を更新します。保存先を明示的に残したい場合だけ、`RESULTS_DIR=/任意のパス`を指定してください。

## 評価値

各実行ファイルが出力する`gpu_kernel_ms`を収集し、ウォームアップを除いた平均・最小・最大・中央値・標準偏差・95%信頼区間をCSVへ記録します。これは最初のタスク開始から最後のタスク終了までのGPU上の実行時間です。グラフの棒は平均、エラーバーは平均の95%信頼区間です。

`validation_status`は、KESCO由来ベンチマークでは全反復の数値検証が成功した場合に`PASS`となります。`validation_checks_per_run`と`max_validation_abs_error`も併記します。合成LIGHT/HEAVYカーネルにはアプリケーション固有の正解値がないため`NOT_APPLICABLE`とします。検証はタイミング計測終了後に実行され、`gpu_kernel_ms`には含まれません。

KESCO由来の6件は、公式grCUDAベンチマークスイートのB1/B5/B6/B7/B8/B10 CUDAカーネルを実行します。合成STGの5件は従来のLIGHT/HEAVY合成CUDAカーネルを実行します。どちらも同じタスクグラフとカーネル入力を4手法で共有し、スケジューリング手法だけを比較します。

SM使用率は実行時間と別にNsight Systemsで測定します。ワークスペース初期化カーネルは除外し、対象タスクの最初のCUDAカーネル開始から最後の終了までの`SMs Active`を集計します。

```bash
cd /home/kobayashi/main/master_study/benchmarks
./run_sm_utilization.sh
```

結果は`../../results/method_comparison_figures/sm_utilization/`にCSV、Nsight Systemsレポート、実カーネル用・STG用の比較グラフとして保存します。
