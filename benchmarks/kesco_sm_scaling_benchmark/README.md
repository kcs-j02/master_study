# KESCO input SM scaling

KESCO由来6入力について、Green Contextで割当SM数を変更し、ベンチマーク全体と各CUDAカーネルの処理時間を測定します。

- SM数: H100で実現可能な`8, 16, ..., 112, 114`
- 入力: 通常評価と同じ固定サイズ・決定的生成データ
- 既定値: 5回ウォームアップ、21回測定
- 時間: CUDA Event時間の中央値
- 初期化: 各反復前に入力状態を復元し、測定区間から除外
- 実行: 全タスクを1本の制限付きStreamでトポロジカル順に実行

```bash
cd /home/kobayashi/main/master_study/benchmarks/kesco_sm_scaling_benchmark
./run.sh
```

短時間の確認:

```bash
WARMUP_RUNS=1 RUNS=2 ./run.sh
```

結果は`../../results/method_comparison_figures/kesco_sm_scaling/`へ保存します。

- `kesco_sm_scaling.csv`: 全測定値の集計
- `kesco_benchmark_sm_scaling.png`: 入力ごとの絶対時間
- `kesco_benchmark_sm_slowdown.png`: 114 SM比
- `<benchmark>_kernel_sm_scaling.png`: カーネル別曲線
- `stage4_prediction_vs_all_benchmarks.png`: 旧Stage 4共通モデルと実測倍率
- `stage4_revised_prediction_vs_measured.png`: 改訂したカーネル別モデルと実測倍率

## Proposed Stage 4への反映

`proposed_method/04_kesco_sm_time_model.hpp`には、この測定で得た8～64 SMの
カーネル別median時間を記録しています。Stage 4はKESCO由来タスクについて、
この実測値を次の規則で使用します。

- 同じカーネルが複数ある場合は、タスクID間のmedianを使用
- 8 SM刻みの測定点は実測時間を直接使用
- 測定点の間は線形補間
- 64 SM以上は64 SMの時間で飽和
- 合成STGは実測プロファイルがないため従来のblock-passモデルを使用

このモデルは、現在の固定入力、64 block構成、測定GPUに対するオフライン
プロファイルです。同じデータから作ったモデルと測定値の一致はモデル実装の
確認であり、未知の入力や別GPUに対する予測精度の検証ではありません。
