

# 修士研究概要

## 概要

### テーマ

**依存タスクの重要度に基づくストリーム割当てとSM配分を用いた単一GPU向け静的スケジューリング・資源割当て手法**

タスクの依存関係から重要度を算出し，

- CUDA Streamへのタスク割当て
- Streaming Multiprocessor（SM）の資源配分
- CUDA Green ContextによるGPU資源分離

を組み合わせることで，全体実行時間の短縮を目指す．

## 提案手法

提案手法では，以下の流れでスケジューリングを行う．

1. タスクグラフの依存関係を解析
2. タスクをレベル化
3. 各タスクから出口タスクまでの最長後続経路長を計算
4. 最長後続経路長（bottom level）をタスク重要度として設定
5. 重要度の高いタスクからStreamへ割当て
6. SM数と予測完了時刻から割当て先を決定
7. 重要タスクを実行するStreamへSMを重点配分
8. CUDA Green ContextによりGPU資源を分離
9. 複数のStream数について予測makespanを比較
10. makespanが最小となる構成を採用

## 使用技術

- C++
- CUDA
- CUDA Toolkit 13.1
- CUDA Stream
- CUDA Green Context
- Taskflow
- cudaFlow
- CMake
- Nsight Systems
- Nsight Compute
- CUDA Events

## 実行環境

- GPU：NVIDIA H100 PCIe
- GPUメモリ：約80 GiB
- CPU：AMD EPYC 7313 16-Core Processor × 2
- メモリ：125 GiB
- OS：Linux

## ディレクトリ構成

### ルートディレクトリ

- `README.md`：プロジェクト説明
- `sample_mixed_chain_parallel.stg`：Critical Pathと並列タスクが混在するSTG
- `sample_fully_parallel.stg`：完全並列型STG
- `sample_Multiple_long_branches.stg`：複数の長い経路を持つSTG
- `sample_random.stg`：依存関係と実行時間が不均一なSTG
- `sample_trial.stg`：ほぼ逐次的なSTG
- `stg_to_png.py`：STG可視化スクリプト
- `run_method_comparison.py`：全手法・全STGの自動比較スクリプト

### `STG/`

Baseline実装．

- `main.cu`：基準方式
- `bench_timer.hpp`：実行時間計測

### `STG_existing_method/`

既存スケジューリング方式．

- `main.cu`：実行本体
- `common_types.hpp`：共通型定義
- `task_DFG_construction.hpp`：依存グラフ構築
- `task_levelization.hpp`：レベル化
- `task_assignment.hpp`：タスク割当て
- `resource_allocation.hpp`：資源割当て

### `STG_existing_method_GC/`

既存方式にCUDA Green ContextによるSM分割を追加した方式．  
各StreamへのSM分割は均等に行う．

### `STG_my_method/`

提案方式．

- `main.cu`：実行本体
- `common_types.hpp`：共通型定義
- `task_DFG_construction.hpp`：依存グラフ構築
- `task_levelization.hpp`：レベル化
- `task_assignment.hpp`：重要度ベースのStream割当て
- `resource_allocation.hpp`：重要度と処理量に基づくSM配分

### その他

- `*.nsys-rep`：Nsight Systemsのプロファイル結果
- `main`：ビルド済み実行ファイル

## 比較手法

以下の4手法を比較する．

1. **Baseline**  
   すべてのタスクを逐次実行する．

2. **Existing**  
   既存手法により複数Streamへタスクを割り当てる．

3. **Existing + GC**  
   ExistingのStream割当てにCUDA Green Contextを適用し，各StreamへSMを均等配分する．

4. **Proposed**  
   bottom levelに基づいてタスク重要度を求め，重要度を考慮したStream割当てとSM配分を行う．

## 評価用STG

### `sample_mixed_chain_parallel.stg`

Critical Pathと多数の並列タスクが混在する構造．  
本研究で主に対象とするSTG．

### `sample_fully_parallel.stg`

多数の独立タスクを持つ完全並列型．

### `sample_Multiple_long_branches.stg`

複数の長い経路を持つ構造．

### `sample_random.stg`

依存関係と実行時間が不均一な構造．

### `sample_trial.stg`

ほぼ逐次的な依存構造．

## 実行方法

### 前提

- CUDA環境が利用可能であること
- Taskflowが利用可能であること
- Taskflowヘッダを`-I`で指定できること

例：

```bash
-I/home/kobayashi/taskflow
````

## 1. Baseline

```bash
cd /home/kobayashi/main/master_study/STG

chmod +x run_batch_stgs.sh
./run_batch_stgs.sh
```

## 2. Existing

```bash
cd /home/kobayashi/main/master_study/STG_existing_method

chmod +x run_batch_stgs.sh
./run_batch_stgs.sh
```

## 3. Existing + GC

```bash
cd /home/kobayashi/main/master_study/STG_existing_method_GC

chmod +x run_batch_stgs.sh
./run_batch_stgs.sh
```

Existing + GCでは，CUDA Green Contextを用いてSMを分割し，各Streamへ均等にSMを割り当てる．

## 4. Proposed

```bash
cd /home/kobayashi/main/master_study/STG_my_method

chmod +x run_batch_stgs.sh
./run_batch_stgs.sh
```

## 一括評価

全5種類のSTGについて，4手法を自動でビルド・実行する．

```bash
cd /home/kobayashi/main/master_study

python3 run_method_comparison.py
```

評価は2段階で実行する．

## Phase 1：実行時間・Speedup

まずNsight Systemsを使用せず，各STG・各手法を通常実行する．

各条件について12回実行し，

* 最初の2回：ウォームアップ
* 残り10回：評価値として使用

とする．

実行時間には，

```text
gpu_submit_wait_ms
```

を使用する．

これはGPU処理を開始してから，全GPU処理が終了するまでの時間を表す．

Speedupは以下で計算する．

```text
Speedup = Baseline実行時間 / 各手法の実行時間
```

Speedupが1より大きい場合，Baselineより高速である．

### 出力例

* `sample_mixed_chain_parallel_gpu_submit_wait.png`
* `sample_mixed_chain_parallel_speedup.png`
* `all_stg_execution_time_comparison.png`
* `all_stg_speedup_comparison.png`

## Phase 2：SMs Active測定

実行時間とSpeedupの測定終了後，全5種類のSTG・全4手法をもう一度実行する．

この実行ではNsight Systemsを用いて，

```text
SMs Active [%]
```

を測定する．

実行時間測定とは分離することで，Nsight Systemsによるプロファイリングの影響を実行時間評価に含めない．

### Nsight Systems実行例

```bash
mkdir -p "$HOME/tmp/nsys"

TMPDIR="$HOME/tmp/nsys" nsys profile \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda \
  --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=10000 \
  -o stg_profile \
  ./main ../sample_mixed_chain_parallel.stg
```

測定結果はNsight SystemsのSQLite形式へ変換し，`SMs Active`のサンプル値を取得する．

GPU処理開始から終了までの区間について平均値を求め，各手法のSM稼働状況を比較する．

### 出力

```text
all_stg_sm_active_comparison.png
```

## 評価指標

### 実行時間

```text
gpu_submit_wait_ms
```

全GPU処理が完了するまでの時間．
小さいほど高速である．

### Speedup

Baselineに対する高速化率．

```text
Speedup = Baseline / Method
```

* `Speedup > 1`：Baselineより高速
* `Speedup = 1`：Baselineと同程度
* `Speedup < 1`：Baselineより低速

### SMs Active

Nsight SystemsのGPU Metricsから取得するSM稼働指標．

GPU処理中にSMがどの程度稼働しているかを評価するために使用する．

## SM割当率との違い

以前使用していた，

```text
total_allocated_sm / available_sm
```

は実際のSM稼働率ではない．

これはGreen Contextなどによって各Streamへ割り当てたSM数の割合を表す**SM割当率**である．

そのため，GPU使用状況の評価にはNsight Systemsから取得した`SMs Active [%]`を使用する．

一方で，

* `total_allocated_sm`
* `sm_stream0`
* `sm_stream1`
* `sm_stream2`
* `sm_stream3`
* `sm_stream4`

は，各方式がどのようにSMを配分したかを確認するための補助情報として利用できる．

## 評価の考え方

本研究の主目的は，単純にSMs Activeを最大化することではなく，依存関係を考慮して重要なタスクへGPU資源を配分し，全体完了時間を短縮することである．

そのため，

* 実行時間
* Speedup
* SMs Active

を組み合わせて評価する．

SMs Activeが高い手法が，必ずしも最短の実行時間になるとは限らない．

Critical Path上の重要タスクへSMを重点的に配分することで，平均SMs Activeが他手法と同程度または低い場合でも，全体完了時間を短縮できる可能性がある．

## Green ContextのSM配分

### Existing + GC

各StreamへSMを**均等配分**する．

例：

```text
利用可能SM = 112
Stream数 = 4

Stream 0 : 28 SM
Stream 1 : 28 SM
Stream 2 : 28 SM
Stream 3 : 28 SM
```

### Proposed

各Streamに含まれるタスクの重要度と処理量を考慮してSMを配分する．

重要度にはbottom levelを使用する．

ストリーム`s`の処理量を，

```text
Ws = Stream sに含まれるタスクの実行時間の総和
```

ストリームの重要度を，

```text
Bs = Stream sに含まれる最大bottom level
```

として，これらを組み合わせてSM配分を決定する．

そのため，Existing + GCとは異なり，各StreamへのSM数は均等ではない．

## Green Context使用時の注意

`STG_existing_method_GC`および`STG_my_method`ではCUDA Green Context APIを使用する．

環境によっては，

```text
cudaDevSmResourceSplit
```

などのAPI呼び出しでエラーが発生する可能性がある．

また，

```text
not enough SMs for evenly divided Green Contexts
```

が表示された場合は，

* Stream数
* `unit_sm`
* `min_group_sm`
* Green ContextのSM分割単位

を確認する．

ログには以下のような情報が出力される．

```text
available SM : <N>
unit SM      : <M>
allocated SM : <K>
unused SM    : <L>
stream[i] SM=<X> [GC]
```

これらは実際のSM稼働率ではなく，各Green Contextへ割り当てたSM資源量を確認するための値である．

## 主な生成ファイル

一括評価を実行すると，以下のようなPNGが生成される．

### 全STG比較

```text
all_stg_execution_time_comparison.png
all_stg_speedup_comparison.png
all_stg_sm_active_comparison.png
```

### STGごとの比較

例：

```text
sample_mixed_chain_parallel_gpu_submit_wait.png
sample_mixed_chain_parallel_speedup.png
```

他のSTGについても同様のファイルが生成される．

```
```
