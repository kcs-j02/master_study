# STG_existing_method

Green Contextを使わない既存手法の比較評価用コードである。

処理順序:

1. STGファイルを読み込む。
2. 依存グラフ（DFG）を構築する。
3. タスクをレベル化する。
4. stream数を最大5本で決定する。
5. 全streamを同じ性能として、既存手法の規則でタスクを割り当てる。
6. Green Contextを作らず、通常CUDA streamで実行する。

全通常streamはGPU資源を共有し、streamごとの専有SM分割は行わない。
