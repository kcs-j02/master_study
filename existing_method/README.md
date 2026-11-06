# existing_method

Green Contextを使わない既存手法の比較評価用コードである。STGをDFGへ
変換して階層化し、依存制約を満たすready-listを階層ごとに処理する。

## 処理の流れ

1. STGを読み込み、タスクの依存グラフ（DFG）を構築する。
2. 入次数0のタスクを同じ階層へ入れ、後続タスクの入次数を減らして階層化する。
3. 最大階層幅と上限5本の小さい方をstream数とする。
4. ready-listからタスクINDEXが小さい順に選ぶ。
5. 依存元のないタスクはINDEX順に各streamへラウンドロビン配置する。
6. 依存元があるタスクは依存元と同じstreamへ配置する。複数の依存元が
   異なるstreamにある場合は、INDEXが最小の依存元のstreamを使用する。
7. 配置したタスクによって入次数が0になった後続タスクをready-listへ追加する。
8. 通常CUDA streamで実行し、GPU資源はstream間で共有する。

この既存手法は、`proc_time`、予測終了時刻、bottom levelをstream選択に
使用しない。bottom levelによる重要度付けは提案手法だけが使用する。

## Stream数

```text
S = min(最大階層幅, 5)
```

複数のstream数候補やSM配分候補は比較しない。

## Existing + GCとの違い

`existing_method_green_context`は同じタスク配置を使用し、配置確定後に
各streamへGreen ContextのSMをほぼ均等に割り当てる。SM数に基づく
タスク再配置や重要度に基づくSM再配分は行わない。
