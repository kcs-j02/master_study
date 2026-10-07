# 単一GPU向けタスクスケジューリング手法の評価

## 研究概要

依存関係を持つGPUタスクを対象に、CUDA Streamへのタスク配置とStreaming Multiprocessor（SM）の資源配分を組み合わせ、全体実行時間の短縮を目指す研究です。

次の4手法を同じベンチマーク入力で比較します。

| フォルダ | 手法 | 概要 |
|---|---|---|
| `sequential_method/` | Sequential | 全タスクを逐次実行する基準手法 |
| `existing_method/` | Existing | bottom levelを用いて複数の通常CUDA Streamへ配置 |
| `existing_method_green_context/` | Existing + GC | Existingと同じ配置にGreen Contextの均等SM配分を適用 |
| `proposed_method/` | Proposed | タスク重要度による配置と、固定配置に対するSM配分探索を実施 |

## 評価の実行方法

### 全ベンチマークを一括評価

一括評価コードは`benchmarks/`直下にあります。

```bash
cd /home/kobayashi/main/master_study/benchmarks
./run_all_benchmarks.sh
```

11個の実行可能なベンチマークを4手法で評価し、44組の結果を1つの統合CSVへまとめます。比較グラフは、合成STGとその他のベンチマークに分けて出力します。

既定では各組合せを12回実行し、先頭2回をウォームアップとして集計から除外します。実行回数は環境変数で変更できます。

```bash
RUNS=20 WARMUP_RUNS=5 ./run_all_benchmarks.sh
```

短時間の動作確認例：

```bash
RUNS=2 WARMUP_RUNS=1 ./run_all_benchmarks.sh
```

### 1つのベンチマークを評価

`KESCO/`と`STG/`以下の各`BENCHMARK_.../`フォルダに、そのベンチマークを4手法で評価する`run_evaluation.sh`があります。

例としてVector Squareを評価する場合：

```bash
cd /home/kobayashi/main/master_study/benchmarks/KESCO/BENCHMARK_vector_square
./run_evaluation.sh
```

HITSを評価する場合：

```bash
cd /home/kobayashi/main/master_study/benchmarks/KESCO/BENCHMARK_hits
RUNS=20 WARMUP_RUNS=5 ./run_evaluation.sh
```

どのベンチマークも同じ方法で実行できます。

```text
KESCO/ または STG/
└── BENCHMARK_<ベンチマーク名>/
    ├── <ベンチマーク名>.stg
    └── run_evaluation.sh
```

### 評価結果

最新の比較CSVとグラフは、毎回同じ名前で次のフォルダへ出力されます。

```text
/home/kobayashi/main/master_study/results/method_comparison_figures/
├── RESULT_stg/                           # benchmarks/STG/ の結果
│   ├── all_stg_results.csv
│   ├── all_stg_execution_time_comparison.png
│   └── <STG名>_execution_time_comparison.png
└── result_BENCHMARK/                    # その他のベンチマーク結果
    ├── all_benchmark_results.csv
    ├── all_benchmark_execution_time_comparison.png
    └── <ベンチマーク名>_execution_time_comparison.png
```

一括評価の比較グラフは、毎回次の2ファイルを更新します。

```text
/home/kobayashi/main/master_study/results/method_comparison_figures/RESULT_stg/all_stg_execution_time_comparison.png
/home/kobayashi/main/master_study/results/method_comparison_figures/result_BENCHMARK/all_benchmark_execution_time_comparison.png
```

最新のCSVとログは次のフォルダへ保存され、次回実行時に同じ場所を更新します。

```text
/home/kobayashi/main/master_study/results/benchmark_evaluation/
```

一括評価の出力：

```text
latest-all/
├── sequential_method/                   # CSVと実行ログ
├── existing_method/
├── existing_method_green_context/
├── proposed_method/
└── all_results.csv                      # 11ベンチマーク×4手法
```

個別評価の出力：

```text
latest-<ベンチマーク名>/
├── sequential_method/
├── existing_method/
├── existing_method_green_context/
├── proposed_method/
├── all_results.csv                      # 1ベンチマーク×4手法
└── <ベンチマーク名>_method_comparison.png
```

## ディレクトリ構成

```text
master_study/
├── sequential_method/                   # Sequentialのアルゴリズム
├── existing_method/                     # Existingのアルゴリズム
├── existing_method_green_context/       # Existing + GCのアルゴリズム
├── proposed_method/                     # Proposedのアルゴリズム
├── benchmarks/
│   ├── run_all_benchmarks.sh            # 全ベンチマーク一括評価
│   ├── KESCO/                            # KESCO由来ベンチマーク
│   │   ├── BENCHMARK_vector_square/
│   │   │   ├── vector_square.stg
│   │   │   └── run_evaluation.sh
│   │   ├── BENCHMARK_black_scholes/
│   │   ├── BENCHMARK_machine_learning/
│   │   ├── BENCHMARK_hits/
│   │   ├── BENCHMARK_image_processing/
│   │   ├── BENCHMARK_deep_learning/
│   │   ├── BENCHMARK_micro_1/            # DAG未公開のため説明のみ
│   │   └── BENCHMARK_micro_2/            # DAG未公開のため説明のみ
│   ├── STG/                              # 合成STGベンチマーク
│   │   ├── BENCHMARK_synthetic_fully_parallel/
│   │   ├── BENCHMARK_synthetic_mixed_chain_parallel/
│   │   ├── BENCHMARK_synthetic_multiple_long_branches/
│   │   ├── BENCHMARK_synthetic_random_dag/
│   │   └── BENCHMARK_synthetic_sequential_chain/
│   ├── benchmark_evaluation/             # 共通評価処理
│   ├── benchmark_inputs/                 # 旧パス互換リンク
│   ├── common/                           # 共通ヘッダ
│   ├── sm_scaling_benchmark/             # SMスケーリング検証
│   └── tools/                            # 補助ツール
├── results/                              # CSV・グラフ・測定結果
└── README.md
```

`benchmark_inputs/`には入力の実体を重複配置せず、以前のパスを使用するコードとの互換性を保つシンボリックリンクだけを置いています。

## 評価対象ベンチマーク

### 実アプリケーション由来

| フォルダ | ベンチマーク |
|---|---|
| `KESCO/BENCHMARK_vector_square/` | Vector Square |
| `KESCO/BENCHMARK_black_scholes/` | Black & Scholes |
| `KESCO/BENCHMARK_machine_learning/` | Machine Learning |
| `KESCO/BENCHMARK_hits/` | HITS |
| `KESCO/BENCHMARK_image_processing/` | Image Processing |
| `KESCO/BENCHMARK_deep_learning/` | Deep Learning |

### 合成タスクグラフ

| フォルダ | 特徴 |
|---|---|
| `STG/BENCHMARK_synthetic_fully_parallel/` | 多数の独立タスクを持つ完全並列型 |
| `STG/BENCHMARK_synthetic_mixed_chain_parallel/` | 長い依存チェーンと並列タスクが混在 |
| `STG/BENCHMARK_synthetic_multiple_long_branches/` | 複数の長い依存経路を持つ |
| `STG/BENCHMARK_synthetic_random_dag/` | 依存関係と処理時間が不均一 |
| `STG/BENCHMARK_synthetic_sequential_chain/` | ほぼ逐次的な依存構造 |

Micro-1とMicro-2は公開資料に元のDAGがないため、`KESCO/BENCHMARK_micro_1/unavailable.md`と`KESCO/BENCHMARK_micro_2/unavailable.md`に理由を記録し、自動評価対象から除外しています。

## 評価指標

評価コードは各実行ファイルが出力する次の値を収集します。

```text
gpu_submit_wait_ms
```

これはGPUへ処理を投入してから、対象となるGPU処理が完了するまでの時間です。ウォームアップを除いた値から、次をCSVへ保存します。

- 平均実行時間
- 最小実行時間
- 最大実行時間
- 実行回数
- ウォームアップ回数

比較グラフの棒は平均値、エラーバーは最小値から最大値の範囲を表します。

現在の4実装が実行するのは、各実装内のlight/heavy合成CUDAカーネルです。元アプリケーション全体の性能ではなく、同じタスクグラフを与えた場合のスケジューリング手法を比較します。

## 各手法の概要

### Sequential

依存関係の順序を守りながら、全タスクを1本の実行系列で処理する基準手法です。

### Existing

1. STGを読み込み、依存グラフを構築する。
2. タスクをレベル化する。
3. 出口タスクまでの残り処理時間を表すbottom levelを計算する。
4. bottom levelが大きいreadyタスクを優先する。
5. 予測完了時刻が小さくなる通常CUDA Streamへ配置する。
6. TaskflowとCUDA Eventで依存制約を保って実行する。

使用するStream数は`min(最大レベル幅, 5)`です。全StreamがGPUのSMを共有し、Streamごとの専有SMは設けません。

### Existing + Green Context

Existingと同じタスク配置を使用します。その後、利用可能なSMをCUDA Green Contextの分割粒度に従って各Streamへ可能な限り均等に配分します。

タスク重要度に応じたSM再配分や、SM配分後のタスク再配置は行いません。

### Proposed

提案手法は次の5段階で処理します。

```text
Stage 1: タスクグラフ解析
    ↓
Stage 2: タスク重要度の計算
    ↓
Stage 3: CUDA Streamへのタスク配置
    ↓
Stage 4: SM配分と実行構成の選択
    ↓
Stage 5: Green Contextを用いたGPU実行
```

#### Stage 1：タスクグラフ解析

`01_task_graph_analysis.hpp`がSTGを読み込み、タスクID、処理時間、先行タスクを解析します。重複ID、存在しない先行タスク、循環依存も検出します。

#### Stage 2：タスク重要度

`02_task_importance.hpp`がbottom levelを計算します。

```text
BL(i) = p(i) + max(BL(j)),  j ∈ succ(i)
```

後続タスクがない場合は`max(BL(j)) = 0`です。値が大きいタスクほど、出口までの残り処理時間が長い重要タスクとして扱います。

#### Stage 3：Stream配置

`03_stream_scheduling.hpp`がready集合からbottom level最大のタスクを選びます。各Streamについて依存完了後の最初の空き区間を調べ、予測完了時刻が最小になるStreamへ配置します。

```text
EFT(i,s) = EST(i,s) + p(i)
```

この段階ではSM数を使用せず、タスク配置とSM配分を分離します。

#### Stage 4：SM配分

`04_execution_configuration_selection.hpp`がStage 3の配置を固定したまま、各Streamの処理量へSM数を反映して予測makespanを評価します。

```text
W_s(M_s) = Σ p_hat(i,s)(M_s)
C_max(M) = max(W_s(M_s))
```

既定の複数Stream構成では、各Streamに最低8 SMを与え、8 SM単位で配分候補を探索します。1～5 Streamの候補から予測makespanが最小の構成を選び、同値の場合はStream数が少ない構成を優先します。

#### Stage 5：GPU実行

`05_green_context_execution.cuh`がStream 0をprimary context上の通常Stream、Stream 1以降をGreen Context上のStreamとして作成します。TaskflowとCUDA Eventで依存関係を保ち、処理完了後にContext、Stream、Event、GPUメモリを解放します。

## Proposedの主な実行オプション

| 環境変数 | 内容 |
|---|---|
| `STG_MAX_STREAMS=<N>` | 評価する最大Stream数を1～5で指定 |
| `STG_DISABLE_GC=1` | Green Contextを使用せず通常Streamを使用 |
| `STG_TWO_STREAM_GC_SM=<N>` | 2 Stream構成を`[114-N, N]`へ変更 |
| `STG_STREAM_SM_COUNTS=<N0,N1,...>` | StreamごとのSM数を直接指定 |
| `STG_BACKGROUND_CHUNKS=<N>` | 背景タスクのカーネル分割数を指定 |
| `STG_DISABLE_STREAM_PLOT=1` | Stream数別グラフの生成を無効化 |

## 実行環境

- GPU：NVIDIA H100 PCIe
- GPUメモリ：約80 GiB
- CPU：AMD EPYC 7313 16-Core Processor × 2
- メモリ：125 GiB
- OS：Linux
- CUDA Toolkit：13.1
- C++ / CUDA / Taskflow / CUDA Green Context

実行にはCUDA環境、Taskflow、Python 3、Matplotlib、NumPyが必要です。

## 注意事項

- `existing_method_green_context`と`proposed_method`はCUDA Green Context APIを使用します。
- GPUによってSM数、分割粒度、最小分割サイズが異なります。
- `not enough SMs for evenly divided Green Contexts`が表示された場合は、Stream数とSM配分を確認してください。
- ログに表示されるStream別SM数は割当量であり、実際のSM稼働率そのものではありません。
- Nsight Systemsで測定する`SMs Active [%]`と、Green ContextのSM割当率は異なる指標です。
