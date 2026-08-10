# 単一GPU向け静的スケジューリング・資源割当て手法

## 概要

テーマ

「依存タスクの重要度に基づくストリーム割当てとSM配分を用いた
単一GPU向け静的スケジューリング・資源割当て手法」


タスクの依存関係から重要度を算出し、

- CUDA Streamへのタスク割当て
- Streaming Multiprocessor（SM）の資源配分
- CUDA Green ContextによるGPU資源分離

を組み合わせることで、全体実行時間の短縮を目指します。

## 提案手法

提案手法では、以下の流れでスケジューリングを行います。

1. タスクグラフの依存関係を解析
2. タスクをレベル化
3. 各タスクから出口タスクまでの最長後続経路長を計算
4. 最長後続経路長をタスク重要度として設定
5. 重要度の高いタスクからStreamへ割当て
6. SM数と予測完了時刻から割当て先を決定
7. 重要タスクを実行するStreamへSMを重点配分
8. 背景タスクをGreen Contextへ分離

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

### 実行環境

- GPU：NVIDIA H100 PCIe
- GPUメモリ：約80 GiB
- CPU：AMD EPYC 7313 16-Core Processor × 2
- メモリ：125 GiB
- OS：Linux

### ディレクトリ構成

ルート直下

- `README.md`：このプロジェクトの説明書
- `common_sample.stg`：STG入力ファイルの共通サンプル
- `stg_to_png.py`：STGファイルを可視化（SVG出力）するスクリプト
- `plot_compare.py`：比較結果CSVからPNGグラフを生成するスクリプト
- `plot_compare_svg.py`：比較結果CSVからSVGグラフを生成するスクリプト

方式ごとの実装ディレクトリ

- `STG/`：ベースライン実装
	- `main.cu`：実行本体（基準方式）
	- `bench_timer.hpp`：計測ユーティリティ
	- `sample.stg`：入力例
- `STG_existing_method/`：既存方式実装
	- `main.cu`：実行本体（既存方式）
	- `common_types.hpp`：共通型定義
	- `task_DFG_construction.hpp`：依存グラフ構築
	- `task_levelization.hpp`：レベル化
	- `task_assignment.hpp`：タスク割当て
	- `resource_allocation.hpp`：SM配分
- `STG_existing_method_GC/`：既存方式 + Green Context 実装
- `STG_my_method/`：提案方式実装
	- `main.cu`：実行本体（提案方式）
	- `common_types.hpp`：共通型定義
	- `task_DFG_construction.hpp`：依存グラフ構築
	- `task_levelization.hpp`：レベル化
	- `task_assignment.hpp`：重要度ベース割当て
	- `resource_allocation.hpp`：重要タスク重視のSM配分

補足

- `*.nsys-rep`：Nsight Systemsのプロファイル結果
- `main`：ビルド済み実行ファイル（生成物）


## 実行方法

前提

- CUDA環境が利用可能であること
- Taskflowヘッダを `-I` で参照できること（例：`-I/home/kobayashi/taskflow`）

1. ベースライン方式（`STG/`）

```
cd STG
nvcc -O2 -std=c++20 main.cu -I/home/kobayashi/taskflow -o main
./main sample.stg
```

2. 既存方式（`STG_existing_method/`）

```
cd STG_existing_method
nvcc -O2 -std=c++20 main.cu -I/home/kobayashi/taskflow -o main
./main sample.stg
```

3. 提案方式（`STG_my_method/`）

```
cd STG_my_method
nvcc -O2 -std=c++20 main.cu -I/home/kobayashi/taskflow -o main
./main sample.stg
```

4. Nsight Systemsで計測（任意）

```
nsys profile --force-overwrite true --trace=cuda,nvtx,osrt -o stg_profile ./main sample.stg
```

`/tmp` が使えない環境では、ユーザディレクトリ配下に一時領域を指定します。

```
mkdir -p "$HOME/tmp/nsys"
TMPDIR="$HOME/tmp/nsys" nsys profile --force-overwrite true --trace=cuda,nvtx,osrt -o stg_profile ./main sample.stg
```

5. 比較結果の可視化

ルート直下に `compare_data.csv` を置いて実行します。

```
python3 plot_compare.py
python3 plot_compare_svg.py
```

### 比較する際の項目とコード

比較CSV（`compare_data.csv`）の列

- `method`
- `gpu_kernel_ms`
- `total_allocated_sm`
- `sm_stream0`
- `sm_stream1`
- `sm_stream2`
- `sm_stream3`
- `sm_stream4`

CSVテンプレート例

```
method,gpu_kernel_ms,total_allocated_sm,sm_stream0,sm_stream1,sm_stream2,sm_stream3,sm_stream4
baseline,120.5,114,114,0,0,0,0
existing_method,95.2,114,60,54,0,0,0
existing_method_gc,90.8,114,56,40,18,0,0
proposed,82.4,114,64,32,18,0,0
```

可視化スクリプトが出力する比較図

- GPU実行時間比較：`compare_gpu_time.png` / `compare_gpu_time.svg`
- 総SM割当て比較：`compare_total_sm.png` / `compare_total_sm.svg`
- Stream別SM割当て比較：`compare_per_stream_sm.png` / `compare_per_stream_sm.svg`

主要な比較観点

- `gpu_kernel_ms`：小さいほど高速
- `baseline / gpu_kernel_ms`：速度向上率（speedup）
- `total_allocated_sm`：割当て総SM数
- `sm_stream*`：StreamごとのSM配分バランス

