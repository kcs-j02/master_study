

# 修士研究概要

## 概要

### テーマ

**依存タスクの重要度に基づくストリーム割当てとSM配分を用いた単一GPU向け静的スケジューリング・資源割当て手法**

タスクの依存関係から重要度を算出し，

- CUDA Streamへのタスク割当て
- Streaming Multiprocessor（SM）の資源配分
- CUDA Green ContextによるGPU資源分離

を組み合わせることで，全体実行時間の短縮を目指す．

## 既存手法

比較対象とする既存手法（Existing）は，依存関係を満たしながらタスクを複数のCUDA Streamへ静的に割り当て，実行の重なりによって全体完了時間を短縮する手法である．

### 処理の流れ

1. STGファイルを読み込み，タスクの依存グラフ（DFG）を構築
2. 入次数が0のタスクを順に取り除き，タスクをレベル化
3. 最大レベル幅と上限5本から使用するStream数を決定
4. 各タスク自身を含む，出口タスクまでの経路上の処理時間和の最大値（bottom level）を計算
5. 全先行タスクが割当て済みのタスクをready集合へ追加
6. ready集合からbottom levelが大きいタスクを優先して選択
7. 全先行タスクの予測完了時刻の最大値以降にある各Streamの最初の空き区間を調べ，原則として予測完了時刻が最小となるStreamへ割当て．ただし，現在の部分makespanを悪化させない場合はstream 0を優先
8. TaskflowとCUDA Eventを用いてDFGの依存制約と同一Stream内の予測開始時刻順の実行を保証
9. Green Contextを使用せず，複数の通常CUDA Streamでタスクを実行

### 特長と制約

- 依存制約を保ったまま，独立に実行可能なタスクの並行性を利用できる．
- bottom levelにより，出口タスクまでの残り処理時間が長いタスクを優先できる．
- Stream末尾だけでなく，依存待ちで生じる空き区間にもタスクを配置する．
- 割当て時には全Streamを同一性能として扱い，使用するStream数は`min(最大レベル幅, 5)`で決定する．
- 全StreamがGPUのSMを共有し，Streamごとの専有SMは設けない．そのため，Green Contextの作成コストは生じない一方，同時実行タスク間のSM競合を制御できない．
- 通信時間，データ転送時間，および実行中の動的な再スケジューリングは考慮しない．

`STG_existing_method_GC`では，上記と同じタスク割当てを決定した後，全StreamをGreen Context上に作成し，使用可能なSMを分割粒度の範囲で可能な限り均等に配分する．SM配分後のタスク再割当てや，タスク重要度に応じたSM再配分は行わない．

## 提案手法

提案手法では，`main.cu`が以下の5段階を順に呼び出してスケジューリングと実行を行う．

1. **STGを解析**：STGファイルを読み込み，記録された処理時間を予測実行時間としてタスク仕様へ変換し，DFGの構築とレベル化を行う．
2. **重要度を判定**：各タスクから出口タスクまでの最長処理時間であるbottom levelを算出する．
3. **Streamへ配置**：全先行タスクが配置済みのready集合からbottom levelが大きいタスクを選び，各StreamのSM数を考慮して原則として予測完了時刻が最小となるStreamへ配置する．重要タスクは最大SM数の優先Stream（既定表ではStream 0）へ集まりやすくし，背景タスクを他のStreamへ分散する．
4. **実行構成を決定**：SM配分候補ごとにStage 3の配置をシミュレーションし，予測makespanが最小の構成を選択する．候補比較APIは本番の実行経路で使用するが，現行挙動を維持するため，既定では固定表または環境変数で指定されたSM配分を1候補として渡す．複数候補を渡せば，同じAPIで比較できる．
5. **GPU上で実行**：Stream 0をprimary context上の通常Stream，Stream 1以降をGreen Context上のStreamとして作成し，TaskflowとCUDA Eventで依存関係を保って実行する．Green Context側の処理完了後はContextを解放し，SMをprimary context側で再利用できる状態に戻す．

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
- `comparison_figures/`：一括評価で生成する全PNGと集約PDFの保存先

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
各StreamへのSM分割は，分割粒度の範囲で可能な限り均等に行う．

### `STG_my_method/`

提案方式．

- `main.cu`：設定を読み取り，Stage 1からStage 5までを順に呼び出す実行本体
- `01_stg_analysis.hpp`：STG読込み，予測実行時間を含むタスク仕様への変換，DFG構築，レベル化
- `02_task_importance.hpp`：後続関係の検証とbottom levelによるタスク重要度の算出
- `03_stream_assignment.hpp`：ready-list schedulingと予測完了時刻に基づくStream配置
- `04_sm_allocation_comparison.hpp`：SM配分候補の生成・検証，各候補の予測makespan比較，実行構成の決定
- `05_green_context_execution.cuh`：CUDA StreamとGreen Contextの作成，TaskflowによるGPU実行，完了後の資源解放
- `bench_timer.hpp`：各StageおよびGPU実行時間の計測

### その他

- `*.nsys-rep`：Nsight Systemsのプロファイル結果
- `main`：ビルド済み実行ファイル

## 比較手法

以下の4手法を比較する．

1. **Baseline**  
   すべてのタスクを逐次実行する．

2. **Existing**  
   bottom levelと予測完了時刻に基づいてタスクを複数の通常Streamへ割り当て，GPU資源を共有して実行する．

3. **Existing + GC**  
   Existingと同じStream割当てを用い，割当て後にCUDA Green ContextでSMをほぼ均等に分割する．

4. **Proposed**  
   bottom levelに基づくStream割当てと，重要タスク用のStream 0へ多くのSMを残す非対称な固定SM配分を用いる．

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

提案手法の`main`を直接実行すると，終了時にStage 1からStage 5までの
実測時間を秒単位（小数点以下9桁）で表示する．

```bash
./main ../sample_mixed_chain_parallel.stg
```

```text
stage_1_stg_analysis_seconds: ... s
stage_2_task_importance_seconds: ... s
stage_3_stream_placement_seconds: ... s
stage_4_sm_allocation_comparison_seconds: ... s
stage_5_green_context_execution_seconds: ... s
```

`run_batch_stgs.sh`では，各STGについてウォームアップ2回を除いた10回の
段階別平均時間を秒単位で表示する．既存の実行時間評価との互換性を保つため，
`gpu_submit_wait_ms`などの従来のミリ秒出力も維持する．
ルートの`run_method_comparison.py`でも，Proposedの各STGについて同じ
段階別平均時間を表示する．

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

生成する図はすべて`comparison_figures/`へ保存される．

* `comparison_figures/sample_mixed_chain_parallel_gpu_submit_wait.png`
* `comparison_figures/sample_mixed_chain_parallel_speedup.png`
* `comparison_figures/all_stg_execution_time_comparison.png`
* `comparison_figures/all_stg_speedup_comparison.png`
* `comparison_figures/all_method_comparison_figures.pdf`：全生成図をまとめた複数ページPDF

同じコマンドを再実行すると，同名のPNGとPDFを最新結果で上書きする．
各図の手法別配色は，Baselineを青，Existingを橙，Existing + GCを緑，Proposedを赤に統一する．

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
comparison_figures/all_stg_sm_active_comparison.png
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

これはGreen Contextによって各Streamへ専有割当てしたSM数の割合を表す**SM割当率**である．通常CUDA StreamでGPU資源を共有するExistingには，この意味でのStream別SM割当てはない．

そのため，GPU使用状況の評価にはNsight Systemsから取得した`SMs Active [%]`を使用する．

一方で，

* `total_allocated_sm`
* `sm_stream0`
* `sm_stream1`
* `sm_stream2`
* `sm_stream3`
* `sm_stream4`

は，Green Contextを使用する方式がどのようにSMを配分したかを確認するための補助情報として利用できる．Existingで出力されるStream別SM値はスケジューリング用の参照値であり，物理的な専有割当てを表さない．

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

利用可能なSMを分割粒度の倍数へ切り下げ，各Streamへ配分単位数が可能な限り等しくなるように配分する．余った配分単位は，Stream IDが小さい順に1単位ずつ加える．

例：

```text
利用可能SM = 114
分割粒度 = 8 SM
Stream数 = 4

Stream 0 : 32 SM
Stream 1 : 32 SM
Stream 2 : 24 SM
Stream 3 : 24 SM
未使用  :  2 SM
```

### Proposed

#### 現行実装のSM数決定

現行実装では，使用するStream数に応じて，H100（114 SM）向けの非対称なSM配分を固定表からあらかじめ選択する．以下は環境変数による上書きを行わない場合の既定値である．

```text
Stream数  SM配分 [Stream 0, Stream 1, ...]
1           [114]
2           [82, 32]
3           [82, 16, 16]
4           [82, 16, 8, 8]
5           [82, 8, 8, 8, 8]
```

評価用に次の上書きも用意している．これらは実験条件を変えるための機能であり，$W_s$ や $B_s$ からの自動配分ではない．

- `STG_DISABLE_GC`：Green Contextを使わず，全Streamを通常CUDA StreamとしてGPU資源を共有
- `STG_TWO_STREAM_GC_SM=<N>`：2 Stream構成を`[114-N, N]`へ上書き
- `STG_STREAM_SM_COUNTS=<N0,N1,...>`：StreamごとのSM数を直接指定

Stream 0はprimary context上の通常Streamとし，複数Stream構成の既定値では82 SMを残す．Stream 1以降はGreen Context上に作成し，残り32 SMを8 SM単位で可能な限り均等に分配する．ただし，固定表を実行時に自動生成するのではなく，Green Contextの実際の分割粒度と最小SM数はGPUから取得して検証する．実機の粒度が固定表と整合しない場合は実行エラーとなる．

2本以上のStreamを使用する場合について，Stream数を $S$，GC Stream数を $K=S-1$，GC側の総SM数を $M_{GC}=32$，分割粒度を $g=8$ とする．配分可能な単位数 $U$，各GC Streamの基本単位数 $q$，余り $a$ は次式となる．

```text
U = M_GC / g = 4
q = floor(U / K)
a = U mod K
```

先頭の $a$ 本のGC Streamへ $g(q+1)$ SM，残りのGC Streamへ $gq$ SMを割り当てると，上記の固定表と同じ数値になる．Stream 0の82 SMはコード上でも固定値であるが，数値上は114 SMからGC側の32 SMを引いた残余に対応する．

このSM数はタスク割当て前に決定し，タスク $i$ をStream $s$ で実行する場合の予測処理時間 $\hat{p}_{i,s}$ に反映する．

```text
p_hat(i,s) = p_i * min(M_ref, L_i) / min(M_s, L_i)
```

- $p_i$：STGに記載されたタスク $i$ の処理時間
- $M_{ref}$：基準SM数（114）
- $M_s$：Stream $s$ に配分したSM数
- $L_i$：タスク $i$ が並列に利用できるSM数の上限（本実装では64）

`STG_KERNEL_AWARE_COST`を指定した評価では，$p_i$ の代わりに実際のカーネル反復回数に基づく次の重みを使用する．

```text
p_i = work_units_i * 3  (HEAVY)
p_i = work_units_i      (LIGHT)
```

例えば，82 SMと114 SMのStreamはどちらも実効SM数が64であるため，予測処理時間は $p_i$ となる．一方，16 SMのStreamでは $4p_i$，8 SMのStreamでは $8p_i$ と見積もる．

ready集合からはbottom levelが大きいタスクを先に選び，部分makespanを悪化させない場合は最大SM数の優先Streamを選ぶ．既定表ではStream 0が優先Streamとなるため，重要タスクはStream 0へ集まりやすくなり，その他のタスクは小さなGreen Context側も利用する．

#### Streamの重要度と処理量からSM数を決定する（拡張設計）

Streamごとの性質からSM数を直接決める場合は，まずタスクの仮割当てを行う．Stream $s$ へ割り当てられたタスク集合を $T_s$ とし，処理量 $W_s$ と重要度 $B_s$ を次式で求める．

```text
W_s = sum(p_i),             i in T_s
B_s = max(bottom_level_i),  i in T_s
```

$W_s$ はそのStreamが担う総処理量を表し，値が大きいStreamへSMを増やすことで負荷の偏りを緩和できる．$B_s$ はそのStreamに含まれる最重要タスクの緊急度を表し，値が大きいStreamを遅らせないことでクリティカルパスの伸長を抑えられる．

$W_s$ と $B_s$ はスケールが異なるため，それぞれの総和で正規化し，Streamの配分スコア $P_s$ を定義する．

```text
W_norm(s) = W_s / sum(W_r)
B_norm(s) = B_s / sum(B_r)
P_s = lambda * B_norm(s) + (1 - lambda) * W_norm(s)
```

$\lambda$ は重要度と処理量のどちらを重視するかを決める係数であり，$0\leq\lambda\leq1$ とする．$\lambda=1$ なら重要度のみ，$\lambda=0$ なら処理量のみを考慮する．両者を同程度に扱う初期値として $\lambda=0.5$ を用いることができるが，最終的な値は予備実験や感度分析によって決定する必要がある．タスクを持たないStreamは $W_s=B_s=P_s=0$ とし，SM配分対象から除外する．また，正規化の分母が0となる指標は，有効なStream間で均等な値 $1/S_a$ にフォールバックする．

SM配分時は，GPUの使用可能SM数を $M$，Green Contextの分割粒度を $g$，Streamごとの最小SM数を $m_{min}$ とし，次の手順を用いる．$g$ と $m_{min}$ は固定値にせず，GPUから取得した情報を用いて次式で求める．

```text
g = max(2, smCoscheduledAlignment)
m_min = ceil(max(2, minSmPartitionSize) / g) * g
```

1. タスクを持つ全Streamへ $m_{min}$ SMずつ割り当てる．
2. 残りのSMを $g$ SMずつの配分単位に分ける．
3. $P_s/M_s$ が最大のStreamを選び，そのStreamへ $g$ SMを追加する．ここで $M_s$ は現在のStream $s$ のSM数である．
4. 配分単位がなくなるまで3を繰り返す．
5. $g$ SM未満の端数は，Green Contextではない通常Stream 0に残す．

有効なStream数を $S_a$ としたとき，$M<S_a m_{min}$ であれば最小SM数を満たせないため，Stream数を減らすか，その構成を無効とする．$P_s/M_s$ が同値の場合は，$P_s$ が大きいStream，さらに同値ならStream IDが小さいStreamを選ぶことで結果を一意にする．

$P_s/M_s$ を用いることで，配分スコアが高いのにSM数が少ないStreamから優先的にSMを増やせる．すべてのSMを一度に比例配分する場合と異なり，最小SM数と分割粒度を常に満たせる．

また，Stream $s$ の配分上限を $C_s=g\lceil(\max_{i\in T_s}L_i)/g\rceil$ とし，$M_s\geq C_s$ のStreamは追加配分の候補から外す．これにより，分割粒度に必要な端数を除き，予測処理時間を短縮できないSM配分を避ける．全Streamが上限に達した後の残余は通常Stream 0に残すか，未使用とする．

例として，3本のStreamの値が次の場合を考える．

```text
             Stream 0  Stream 1  Stream 2
W_s              60        30        10
B_s             100        40        20
W_norm(s)       0.60      0.30      0.10
B_norm(s)      0.625      0.25     0.125
P_s (lambda=0.5)
               0.6125     0.275    0.1125
```

$M=114$，$g=8$，$m_{min}=8$ とする．まず3本のStreamへ8 SMずつ，合計24 SMを割り当てる．残り90 SMのうち88 SMを11個の配分単位として順に配分すると`[64, 32, 16]`となり，端数2 SMを通常Stream 0に残すため，最終的な配分は次のようになる．

```text
Stream 0 : 66 SM
Stream 1 : 32 SM
Stream 2 : 16 SM
```

Stream 0の66 SMには，8 SM単位で配分した64 SMと，分割粒度未満の端数2 SMが含まれる．SM配分後は，決定した $M_s$ でタスク割当てと予測makespanを再計算する．更新前より予測makespanが短くなる場合のみ新しい構成を採用する．反復する場合は上限回数を設け，予測makespanが改善しない場合または同じ構成が再現した場合に終了する．

> **実装状況：** 現行の`STG_my_method` は上記の $W_s$，$B_s$，$P_s$ によるSM配分と再割当てをまだ実装していない．現在はStream数別の固定表を使用しており，$W_s$ はログ出力にのみ使用し，Stream別の $B_s$ は集計していない．そのため，この拡張設計を提案手法の実装済み機能として評価するには，コードへの組み込みが必要である．

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

一括評価を実行すると，すべてのPNGが`comparison_figures/`に生成される．
さらに，全PNGと同じ図を次の複数ページPDFへまとめる．

```text
comparison_figures/all_method_comparison_figures.pdf
```

### 全STG比較

```text
comparison_figures/all_stg_execution_time_comparison.png
comparison_figures/all_stg_speedup_comparison.png
comparison_figures/all_stg_sm_active_comparison.png
```

### STGごとの比較

例：

```text
comparison_figures/sample_mixed_chain_parallel_gpu_submit_wait.png
comparison_figures/sample_mixed_chain_parallel_speedup.png
```

他のSTGについても同様のファイルが生成される．
再実行時は各ファイルを最新の評価結果で上書きする．
