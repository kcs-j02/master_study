# STG_existing_method_GC

`STG_existing_method` にGreen Contextの均等SM配分だけを追加した
比較評価用コードである。

- タスク選択・stream数決定・stream割り当ては既存手法と同じ。
- タスクのstream割り当てを確定してからGCのSMを均等配分する。
- GCのSM配分結果を使ったタスクの再割り当ては行わない。
- stream 0を含む全実行streamをGreen Context上に作成する。
- 使用可能SMを、GPUの分割粒度の範囲で均等に割り当てる。
- 分割粒度未満の端数SMは使用しない。
- タスク重要度に応じたSM再配分は行わない。

```sh
nvcc -O2 -std=c++20 main.cu -I/home/kobayashi/taskflow -o main
./main sample.stg
```
