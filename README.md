

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

提案手法では，`00_pipeline_configuration.hpp`に共通定数，実行オプション，
およびSM配分候補をまとめる．`main.cu`はその設定を読み取り，以下の5段階で
タスク配置，SM配分の決定，GPU実行を行う．

1. **STGを解析**：STGファイルを読み込み，記録された処理時間を予測実行時間としてタスク仕様へ変換し，DFGの構築とレベル化を行う．
2. **各タスクの重要度を作成**：SM配分とは独立に，各タスクの処理時間と出口タスクまでの最長後続経路の処理時間和からbottom levelを1回だけ算出する．
3. **タスクをStreamへ配置**：各Stream数について，全先行タスクが配置済みのready集合からbottom levelが大きいタスクを選び，元の処理時間とStream内の空き区間を用いて予測完了時刻が最小となるStreamへ配置する．この段階ではSM数を使用しない．
4. **SMを配分**：Stage 3のタスク配置を固定し，各Streamのタスク処理量に対してSM数を反映した予測時間を評価する．既定の複数Stream構成ではSMを8 SM単位で探索・配分し，最終的に予測makespanが最小の構成を選択する．同値の場合はStream数が少ない構成を優先する．
5. **GPU上で実行**：Stream 0をprimary context上の通常Stream，Stream 1以降をGreen Context上のStreamとして作成し，TaskflowとCUDA Eventで依存関係を保って実行する．Green Context側の処理完了後はContextを解放し，SMをprimary context側で再利用できる状態に戻す．

### 提案手法の処理手順（00〜05）

```text
00 共通設定・SM候補
        ↓
01 STG解析
        ↓
02 各タスクの重要度作成
        ↓
03 タスクのStream配置
        ↓
04 SM配分
        ↓
05 GPU実行
```

#### 00．共通設定とSM候補の生成

`00_pipeline_configuration.hpp`は独立した計算Stageではなく，01〜05が共通で使用する設定を準備する．GPU全体の基準SM数を114，最大Stream数を5，1タスクの並列SM上限と`proc_time`の基準SM数を64として定義する．

また，`STG_MAX_STREAMS`，`STG_DISABLE_GC`，`STG_STREAM_SM_COUNTS`などの環境変数を`PipelineOptions`へ変換し，Stage 4に渡すSM候補を生成する．既定候補は`[114]`，`[58,56]`，`[42,40,32]`，`[34,32,24,24]`，`[26,24,24,24,16]`である．既定実行の2〜5 Streamでは，これらの数値を最終配分とするのではなく，主に評価するStream数をStage 3・4へ与える．

- **入力**：環境変数
- **処理**：共通定数の定義，実行オプションの読込み，SM候補の生成と検証
- **出力**：`PipelineOptions`と`vector<vector<int>>`形式のSM候補

#### 01．STGの解析

`01_stg_analysis.hpp`は入力STGを読み込み，各タスクのID，`proc_time`，先行タスクを`TaskSpec`へ変換する．タスクを`proc_time`の中央値に基づいてLIGHT/HEAVYに分類し，GPUカーネルの反復量`work_units`も設定する．

次に，先行関係から後続辺を作成してタスクDFGを構築する．入次数0のタスクからlevel化し，循環依存，重複ID，存在しない先行タスクも検出する．

- **入力**：STGファイルパス
- **処理**：STG読込み，`TaskSpec`作成，LIGHT/HEAVY分類，DFG構築，level化
- **出力**：`StgAnalysisResult`（STG，タスク列，DFG，level列）

#### 02．各タスクの重要度作成

`02_task_importance.hpp`は，タスクの処理時間と後続タスクからbottom level（BL）を計算する．

```text
BL(i) = p(i) + max(BL(j)),  j ∈ succ(i)
```

後続タスクがない場合は`max(BL(j)) = 0`とする．BLが大きいタスクほど，そのタスクから出口タスクまでの残り処理時間が長いため，Stage 3で優先して配置する．BLにはStream数やSM配分を反映せず，全候補に対して1回だけ計算する．

- **入力**：Stage 1の`vector<TaskSpec>`
- **処理**：successor表の構築，DFSによるbottom levelのメモ化計算
- **出力**：`TaskImportanceResult`（successor表とタスクID別bottom level）

#### 03．タスクのStream配置

`03_stream_assignment.hpp`は，SM数を使わずにStream数ごとのタスク配置を作成する．先行タスクがすべて配置済みのタスクをready集合とし，その中からBL最大のタスクを選ぶ．BLが同じ場合は`proc_time`が大きいタスク，それも同じ場合はIDが小さいタスクを優先する．

選択したタスク (i) に対し，各Stream (s) の依存完了時刻以降にある最初の空き区間から予測開始時刻`EST(i,s)`を求め，次式を計算する．

```text
EFT(i,s) = EST(i,s) + p(i)
```

`EFT`が最小のStreamへ配置し，同値なら`EST`，それも同値ならStream IDが小さい方を選ぶ．同じStream数にSM候補が複数あっても，この配置は共通である．

- **入力**：タスク列，`TaskImportanceResult`，Stream数
- **処理**：ready-list scheduling，Stream内空き区間への挿入，EFT最小のStream選択
- **出力**：`StreamScheduleResult`（Stream数，タスク配置，予測開始・終了時刻，配置時makespan）

#### 04．SM配分と最終構成の選択

`04_execution_configuration_selection.hpp`は，Stage 3のタスク配置を固定したまま，各Streamへ配分するSM数を決定する．タスク (i) のSM数補正後の予測処理時間を \(\hat{p}_{i,s}(M_s)\) とし，Stream (s) の負荷と評価makespanを次式で定義する．

```text
W_s(M_s) = Σ p_hat(i,s)(M_s),  task i is assigned to Stream s
C_max(M) = max(W_s(M_s)),          0 ≤ s < Stream数
```

既定の2〜5 Stream構成では，各Streamに最低8 SMを与え，初期SMプールの全配分を8 SMチャンク単位で全探索する．その最良配分を固定し，114 SMを8 SM単位で分けた端数2 SMをStream 0へ加え，残りのSMも8 SM単位で再び全探索する．各段階で \(C_{max}\) が最小の配分を選び，同値なら追加チャンク数の二乗和が小さい配分を選ぶ．

最後に1〜5 Streamの最良候補を比較し，予測makespanが最小の構成を採用する．同値ならStream数が少ない候補を優先する．この評価値はSTGの依存待ちを含む厳密なDAG makespanではなく，固定配置に対する最大Stream負荷である．

##### 予測処理時間の計算例

`proc_time = 1000`，タスクのSM上限`L_i = 64`，要素数`N = 16384`のタスクを考える．基準の64 SMでは，1 block当たり256要素，256 threads，1 passとなるため，予測処理時間は元の`proc_time`と同じ1000である．

| StreamのSM数 $M_s$ | 有効SM数 | 1 block当たり要素数 | threads/block | pass数 | 予測処理時間 |
|---:|---:|---:|---:|---:|---:|
| 114 | 64 | 256 | 256 | 1 | 1000 |
| 64 | 64 | 256 | 256 | 1 | 1000 |
| 32 | 32 | 512 | 256 | 2 | 2000 |
| 24 | 24 | 683 | 256 | 3 | 3000 |
| 16 | 16 | 1024 | 256 | 4 | 4000 |
| 8 | 8 | 2048 | 256 | 8 | 8000 |

比例式を用いて同じ関係を簡略化して表す場合は，次式になる．

```text
p_hat(i,s) = p_i * min(M_proc, L_i)
                   / min(M_s, M_gpu, L_i, M_proc)
```

この式は，「基準となるSM数」と「実際にタスクが使えるSM数」の比で，元の処理時間を拡大する近似である．

- `p_i`は，STGに記録されたタスク $i$ の基準処理時間である．
- 分子の`min(M_proc, L_i)`は，基準処理時間を取得したときの有効SM数を表す．基準SM数が64でも，タスク自身が32 SMまでしか並列化できない場合は32を使う．
- 分母の`min(M_s, M_gpu, L_i, M_proc)`は，Streamの配分SM数，GPU全体のSM数，タスクの並列上限，基準SM数のうち，最も小さい値である．これを実行時の有効SM数とみなす．
- 「基準有効SM数 ÷ 実行時有効SM数」を`p_i`に掛ける．実行時のSM数が基準の半分なら予測処理時間は2倍，4分の1なら4倍になる．
- Streamへ基準以上のSMを配分しても，`L_i`または`M_proc`で打ち切られるため，予測処理時間はそれ以上短くならない．

例えば，`p_i = 1000`，`M_proc = 64`，`L_i = 64`，`M_gpu = 114`とする．Streamへ32 SMを配分した場合は，

```text
p_hat(i,s) = 1000 * min(64, 64)
                    / min(32, 114, 64, 64)
           = 1000 * 64 / 32
           = 2000
```

この例では，基準の64 SMに対して実行時に使えるSMが32である．利用可能なSM数が半分になるため，処理時間は基準の1000から2倍の2000になると予測する．同じ値をほかのSM配分へ代入すると，次のようになる．

| StreamのSM数 $M_s$ | 分子`min(64,64)` | 分母`min(M_s,114,64,64)` | 計算 | $\hat{p}_{i,s}$ |
|---:|---:|---:|---:|---:|
| 114 | 64 | 64 | $1000\times64/64$ | 1000 |
| 82 | 64 | 64 | $1000\times64/64$ | 1000 |
| 64 | 64 | 64 | $1000\times64/64$ | 1000 |
| 32 | 64 | 32 | $1000\times64/32$ | 2000 |
| 24 | 64 | 24 | $1000\times64/24$ | 2666.67 |
| 16 | 64 | 16 | $1000\times64/16$ | 4000 |
| 8 | 64 | 8 | $1000\times64/8$ | 8000 |

この比例式では24 SMの予測値は約2666.67になる．一方，現行コードは上で説明した整数pass数モデルを使用するため，24 SMでは`pass = 3`となり，予測値は3000になる．比例式はSM数と処理時間の関係を連続的に表した近似であり，実装が実際に候補評価へ使用する値はpass数モデルの値である．

例えば，Stage 3の固定配置が次のようになっているとする．

```text
Stream 0: Task A (proc_time=1000), Task B (proc_time=500)
Stream 1: Task C (proc_time=800),  Task D (proc_time=400)
```

Stream 0へ98 SM，Stream 1へ16 SMを配分すると，Stream 0はタスクのSM上限64で打ち切られて1 pass，Stream 1は4 passであるため，各Streamの予測処理時間合計は次のようになる．

```text
W_0(98) = 1000 * 1 + 500 * 1 = 1500
W_1(16) =  800 * 4 + 400 * 4 = 4800

C_max([98,16]) = max(1500, 4800) = 4800
```

この場合はStream 1がボトルネックである．Stage 4はSM配分候補ごとに同じ計算を行い，この`C_max`が最小になる配分を選択する．

- **入力**：タスク列，Stream数ごとの`StreamScheduleResult`，00のSM候補
- **処理**：SM数別タスク時間の予測，Stream負荷評価，二段階の8 SM単位全探索，Stream数間の最終比較
- **出力**：`SmAllocationStageResult`（全候補，Stream数別最良候補，`SmAllocationDecision`）

#### 05．Green Contextを用いたGPU実行

`05_green_context_execution.cuh`は，Stage 4で選択したSM配分とタスク配置をGPU上に構成する．実行前にGPUから総SM数，最小分割サイズ，alignmentを取得し，Green Contextとして実現可能なSM配分かを検証する．

Stream 0はprimary context上の通常CUDA Streamとし，Stream 1以降に対して`cudaDevSmResourceSplit`でSMを切り出し，resource descriptor，Green Context，execution-context Streamを生成する．Taskflowに元のSTG依存と同一Stream内順序を設定し，CUDA Eventによって先行タスクのGPU完了を待ってからLIGHT/HEAVYカーネルを起動する．

Green Context側のStreamで最後のタスクが完了したら，そのStreamとContextを解放し，解放したSMをStream 0側で再利用できる状態に戻す．実行後は資源構築時間，GPU投入から完了までの時間，最初のkernel開始から最後のkernel終了までの時間を計測し，CUDA Event，Stream，Context，GPUメモリを解放する．

- **入力**：タスク列，Stage 4の`SmAllocationDecision`，実行オプション
- **処理**：SM resource分割，Green Context/Stream作成，Taskflow依存実行，完了GCの動的解放，CUDA時間計測，resource cleanup
- **出力**：`GreenContextExecutionResult`（resource setup，GPU submit/wait，kernel区間の各時間）

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
- `00_pipeline_configuration.hpp`：全Stageで共有する定数，実行オプションと環境変数の読込み，SM配分候補の定義・生成・検証
- `01_stg_analysis.hpp`：STG読込み，予測実行時間を含むタスク仕様への変換，DFG構築，レベル化
- `02_task_importance.hpp`：SM配分に依存しないタスク処理時間と後続経路に基づくbottom levelの算出
- `03_stream_assignment.hpp`：bottom levelによるready-list schedulingと，SM配分に依存しないタスクのStream配置
- `04_execution_configuration_selection.hpp`：Stage 3で固定した配置に対するSM配分の探索・評価と，予測makespan最小の最終構成の選択
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
   SM配分に依存しないbottom levelを1回算出し，その重要度に基づいてSM数を使わずにタスクをStreamへ配置する．その配置を固定したままSM配分を最適化し，予測makespanが最小の最終構成を選択する．

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
また，Stage 2の計算後に各タスクの重要度（bottom level）を表示する．

```bash
./main ../sample_mixed_chain_parallel.stg
```

```text
===== Task importance =====
task 0 : importance=...
task 1 : importance=...
===========================

stage_1_stg_analysis_seconds: ... s
stage_2_task_importance_seconds: ... s
stage_3_stream_placement_seconds: ... s
stage_4_sm_allocation_seconds: ... s
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
* `comparison_figures/all_stg_execution_time_sm_active_comparison.png`
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
comparison_figures/all_stg_execution_time_sm_active_comparison.png
```

後者はSTGごとに「GPU Submit Wait Time」と「Average SMs Active」の
散布図・線形回帰線・Pearson相関係数を表示する．

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

現行実装では，1 Streamから`STG_MAX_STREAMS`で指定する上限（既定は5）までを
比較する．`00_pipeline_configuration.hpp`が供給する既定のSM配分候補は次のとおりである．

```text
Stream数  SM配分候補 [Stream 0, Stream 1, ...]
1           [114]
2           [58, 56]
3           [42, 40, 32]
4           [34, 32, 24, 24]
5           [26, 24, 24, 24, 16]
```

2 Stream以上の値はStage 4へ渡す既定候補であり，最適化後の最終SM配分はSTGのタスク配置によって変化する．

Stage 2では，SM配分候補に依存しないbottom levelを全候補に共通の重要度として1回だけ計算する．
Stage 3では各Stream数について，SM配分候補の数値を使わずにタスクのStream配置を計算する．
同じStream数のSM候補が複数あっても，タスク配置は共通である．

Stage 4ではStage 3の配置を固定し，そのStream数に対するSM配分を評価・最適化する．
既定の複数Stream構成では，各Streamに最低8 SMを与えた上で，次の総SM数を8 SM単位で全探索する．

```text
Stream数  第1段階のSMプール  Stream 0へ加える端数  第2段階の追加SM
2         112                     2                         0
3          96                     2                        16
4          96                     2                        16
5          80                     2                        32
```

第1段階でSMプールの配分を決め，分割粒度未満の2 SMをprimary context上のStream 0へ加えた後，
第2段階の残りを再び8 SM単位で配分する．各段階では，SM数で補正した各Streamの処理時間合計の最大値が
最小となる配分を選ぶ．最後に全Stream数の候補を比較し，予測makespanが最小の実行構成を選択する．

実験条件と出力を変更するため，次の環境変数を用意している．
これらは $W_s$ や $B_s$ からSM数を自動配分する機能ではない．

- `STG_MAX_STREAMS=<N>`：評価する最大Stream数を1から5の範囲で指定
- `STG_DISABLE_GC`：Green Contextを使わず，全Streamを通常CUDA StreamとしてGPU資源を共有
- `STG_TWO_STREAM_GC_SM=<N>`：2 Stream構成を`[114-N, N]`へ上書き
- `STG_STREAM_SM_COUNTS=<N0,N1,...>`：StreamごとのSM数を直接指定
- `STG_BACKGROUND_CHUNKS=<N>`：背景タスクのカーネル分割数を1から16の範囲で指定
- `STG_DISABLE_STREAM_PLOT`：Stream数別の比較グラフ生成を無効化

`STG_STREAM_SM_COUNTS`または`STG_TWO_STREAM_GC_SM`を指定した場合は，
指定から得た1候補だけを評価する．`STG_DISABLE_GC`を指定した場合は，
各Stream数について全Streamの参照SM数を114としてStage 4の配分評価を行い，
実行時には通常CUDA Stream間でGPU資源を共有する．

Stream 0はprimary context上の通常Stream，Stream 1以降はGreen Context上のStreamとする．
Stage 4で決定した配分について，Green Contextの実際の分割粒度と最小SM数をGPUから取得して検証する．
実機の条件で実現できない配分は実行エラーとなる．

Stage 2では，SM数を考慮せず，タスク $i$ の処理時間 $p_i$ と後続タスクの
bottom levelから重要度 $BL(i)$ を次式で計算する．後続タスクがない場合の
最大値は0とする．この計算はSM配分候補の評価前に1回だけ行う．

```text
BL(i) = p_i + max(BL(j)),  j in succ(i)
```

Stage 4では，各候補のSM数をタスク $i$ をStream $s$ で実行する場合の
予測処理時間 $\hat{p}_{i,s}$ に反映する．

```text
E_base(i) = min(M_proc, L_i)
E(i,s)    = max(1, min(M_s, M_gpu, L_i))

elements_per_block(E) = ceil(N / E)
threads_per_block(E)  = max(32, min(256,
                               32 * ceil(elements_per_block(E) / 32)))
pass(E) = ceil(elements_per_block(E) / threads_per_block(E))

p_hat(i,s) = p_i * pass(E(i,s)) / pass(E_base(i))
```

- $p_i$：STGに記載されたタスク $i$ の処理時間
- $M_s$：Stream $s$ に配分したSM数
- $M_{gpu}$：GPU側で利用可能なSM数の上限（114）
- $M_{proc}$：proc_timeを取得した基準SM数（64）
- $L_i$：タスク $i$ 自身が並列に利用できるSM数の上限（本実装では64）
- $N$：タスクの要素数（$64\times256=16384$）

Stage 2のbottom levelは上式のStream別予測処理時間を使用せず，全候補で共通となる．
Stage 3は元の処理時間だけを使って配置先を決め，Stage 4がStream別予測処理時間を使ってSM配分と予測makespanを評価する．

`STG_KERNEL_AWARE_COST`を指定した評価では，$p_i$ の代わりに実際のカーネル反復回数に基づく次の重みを使用する．

```text
p_i = work_units_i * 3  (HEAVY)
p_i = work_units_i      (LIGHT)
```

例えば，82 SMと114 SMのStreamはどちらも実効SM数が64であるため，予測処理時間は $p_i$ となる．一方，16 SMのStreamでは $4p_i$，8 SMのStreamでは $8p_i$ と見積もる．

ready集合からはStage 2で算出したbottom levelが大きいタスクを先に選ぶ．Stage 3の配置先は，
元の処理時間から求めた予測完了時刻，予測開始時刻，Stream IDの順で比較して決める．

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

> **実装状況：** 現行の`STG_my_method` は上記の $W_s$，$B_s$，$P_s$ によるSM配分とタスクの再割当てを実装していない．現在はStage 3のタスク配置を固定し，Stage 4で各Streamの処理時間合計を基に8 SM単位の配分を全探索する．Stream別の $B_s$ は集計していないため，この拡張設計を提案手法の実装済み機能として評価するには，コードへの組み込みが必要である．

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
comparison_figures/all_stg_execution_time_sm_active_comparison.png
```

### STGごとの比較

例：

```text
comparison_figures/sample_mixed_chain_parallel_gpu_submit_wait.png
comparison_figures/sample_mixed_chain_parallel_speedup.png
```

他のSTGについても同様のファイルが生成される．
再実行時は各ファイルを最新の評価結果で上書きする．
