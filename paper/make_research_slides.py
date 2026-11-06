from pathlib import Path

from PIL import Image, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "research_summary_slides.pptx"
ASSETS = ROOT / "paper" / "slide_assets"
ASSETS.mkdir(exist_ok=True)

W, H = 13.333, 7.5
BLACK = RGBColor(20, 20, 20)
DARK = RGBColor(55, 55, 55)
GRAY = RGBColor(105, 105, 105)
MID = RGBColor(185, 185, 185)
LIGHT = RGBColor(242, 242, 242)
WHITE = RGBColor(255, 255, 255)


def gray_image(relative: str) -> str:
    source = ROOT / relative
    target = ASSETS / source.name
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        image = Image.open(source).convert("RGB")
        ImageOps.grayscale(image).save(target)
    return str(target)


prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)


def text(slide, x, y, w, h, value, size=18, color=BLACK, bold=False,
         align=PP_ALIGN.LEFT, font="Noto Sans CJK JP"):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.04)
    frame.margin_right = Inches(0.04)
    frame.margin_top = Inches(0.02)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def box(slide, x, y, w, h, fill=WHITE, line=MID, rounded=True):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if rounded else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1)
    return shape


def line(slide, x1, y1, x2, y2, color=BLACK, width=1.5, arrow=False):
    connector = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    connector.line.color.rgb = color
    connector.line.width = Pt(width)
    if arrow:
        connector.line.end_arrowhead = True
    return connector


def image(slide, relative, x, y, w=None, h=None):
    path = gray_image(relative)
    kwargs = {}
    if w is not None:
        kwargs["width"] = Inches(w)
    if h is not None:
        kwargs["height"] = Inches(h)
    slide.shapes.add_picture(path, Inches(x), Inches(y), **kwargs)


def header(slide, section, heading, number):
    line(slide, 0, 0.16, W, 0.16, BLACK, 4)
    text(slide, 0.55, 0.35, 2.0, 0.25, section.upper(), 10, GRAY, True)
    text(slide, 0.55, 0.68, 12.0, 0.55, heading, 26, BLACK, True)
    text(slide, 12.15, 7.08, 0.6, 0.18, str(number), 9, GRAY, False, PP_ALIGN.RIGHT)


def footer(slide):
    text(slide, 0.55, 7.08, 5.0, 0.18, "Static Task Graph / CUDA Stream scheduling", 9, GRAY)


def slide(section, heading):
    current = prs.slides.add_slide(prs.slide_layouts[6])
    current.background.fill.solid()
    current.background.fill.fore_color.rgb = WHITE
    header(current, section, heading, len(prs.slides))
    footer(current)
    return current


def bullet(slide, x, y, w, value, size=16):
    text(slide, x, y, 0.2, 0.3, "■", 10, BLACK, True)
    text(slide, x + 0.25, y, w - 0.25, 0.38, value, size, BLACK)


# 1. Cover
s = prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid()
s.background.fill.fore_color.rgb = WHITE
line(s, 0, 0.18, W, 0.18, BLACK, 5)
text(s, 0.8, 1.15, 11.8, 0.4, "研究発表", 14, GRAY, True)
text(s, 0.8, 1.75, 11.8, 1.0, "単一GPU上のStatic Task Graph実行における\nCUDA Stream配置とSM資源配分", 31, BLACK, True)
text(s, 0.83, 3.35, 10.8, 0.5, "実カーネルの依存関係・代表処理時間・SMプロファイルを用いた二段階ヒューリスティック", 16, DARK)
box(s, 0.82, 4.65, 11.7, 1.05, LIGHT, MID)
text(s, 1.1, 4.92, 11.1, 0.45, "STG解析  →  重要度に基づくStream配置  →  実測SMモデルによる資源配分  →  実GPU検証", 17, BLACK, True, PP_ALIGN.CENTER)
text(s, 0.83, 6.55, 5, 0.22, "KESCO由来アプリケーション6種 / 合成STG", 11, GRAY)
text(s, 12.15, 7.08, 0.6, 0.18, "1", 9, GRAY, False, PP_ALIGN.RIGHT)

# 2. Background
s = slide("01 / 背景", "なぜStream数を増やすだけでは不十分か")
box(s, 0.7, 1.95, 3.6, 3.9, LIGHT, MID)
text(s, 0.98, 2.25, 3.0, 0.4, "依存関係", 20, BLACK, True)
text(s, 0.98, 2.85, 3.0, 0.7, "独立タスクは並列化できるが、\n後続タスクは先行完了を待つ", 16, DARK)
box(s, 4.85, 1.95, 3.6, 3.9, LIGHT, MID)
text(s, 5.13, 2.25, 3.0, 0.4, "SM競合", 20, BLACK, True)
text(s, 5.13, 2.85, 3.0, 0.7, "複数Streamは同じGPU資源を共有し、\nカーネル同士がSMを奪い合う", 16, DARK)
box(s, 9.0, 1.95, 3.6, 3.9, LIGHT, MID)
text(s, 9.28, 2.25, 3.0, 0.4, "実カーネル差", 20, BLACK, True)
text(s, 9.28, 2.85, 3.0, 0.7, "カーネルごとにSM数への感度が異なり、\n一様なコストモデルでは扱えない", 16, DARK)
text(s, 0.75, 6.2, 11.8, 0.45, "実行時間は、DAGの並列性・クリティカルパス・SM競合・投入／同期コストの組合せで決まる。", 18, BLACK, True, PP_ALIGN.CENTER)

# 3. Research question
s = slide("02 / 目的", "研究目的と検証する問い")
text(s, 0.8, 1.95, 11.5, 0.45, "DAG構造と実カーネル特性を使うことで、Stream配置とSM配分をどこまで改善できるか？", 21, BLACK, True)
questions = [
    ("Q1", "Stream並列性", "独立タスクを複数Streamへ配置すると、実アプリケーションの時間は短くなるか？"),
    ("Q2", "重要度", "bottom levelによる優先順位付けは、既存配置へ追加効果を持つか？"),
    ("Q3", "SM配分", "実測したSM数別カーネル時間による資源配分は有効か？"),
]
for i, (label, title, detail) in enumerate(questions):
    y = 2.75 + i * 1.15
    box(s, 0.85, y, 1.05, 0.75, BLACK, BLACK)
    text(s, 0.85, y + 0.16, 1.05, 0.35, label, 17, WHITE, True, PP_ALIGN.CENTER)
    text(s, 2.15, y + 0.05, 2.3, 0.3, title, 18, BLACK, True)
    text(s, 4.25, y + 0.05, 7.8, 0.5, detail, 15, DARK)
text(s, 0.85, 6.35, 11.5, 0.35, "仮説：Stream配置の効果は独立分岐が多いDAGで現れやすいが、重要度とSM配分の効果は一様ではない。", 15, BLACK, True)

# 4. Real kernel -> STG
s = slide("03 / 入力モデル", "実カーネルをSTGへ対応づける方法")
steps = [
    ("1", "カーネル確認", "task IDごとのCUDAカーネルと入出力バッファを確認"),
    ("2", "依存関係定義", "データを生成するタスクを先行タスクとして列挙"),
    ("3", "時間測定", "固定入力・launch条件で代表処理時間を測定"),
    ("4", "SM測定", "8〜64 SMで実行し、カーネル別プロファイルを作成"),
]
for i, (num, title, detail) in enumerate(steps):
    x = 0.75 + (i % 2) * 6.1
    y = 2.0 + (i // 2) * 1.75
    box(s, x, y, 5.5, 1.25, LIGHT, MID)
    text(s, x + 0.2, y + 0.2, 0.45, 0.35, num, 18, BLACK, True)
    text(s, x + 0.8, y + 0.15, 4.3, 0.3, title, 18, BLACK, True)
    text(s, x + 0.8, y + 0.58, 4.3, 0.45, detail, 13, DARK)
text(s, 0.85, 5.8, 11.5, 0.75, "重要：STGは実行トレースから自動抽出したものではない。\n実装上のtask ID・データ依存関係・測定値を、静的スケジューラの入力形式へ整理する。", 16, BLACK, True)

# 5. Mapping figure
s = slide("03 / 対応関係", "STGノードは実際に起動されるCUDA処理単位である")
image(s, "paper/evidence_figures/kernel_measurement_mapping.png", 0.65, 1.85, w=7.1)
box(s, 8.15, 1.95, 4.5, 3.95, LIGHT, MID)
text(s, 8.45, 2.25, 3.8, 0.35, "対応づけの要点", 19, BLACK, True)
bullet(s, 8.45, 2.9, 3.8, "task IDはSTG・ランチャ・検証で共通")
bullet(s, 8.45, 3.65, 3.8, "proc_timeは実カーネルの代表測定値")
bullet(s, 8.45, 4.4, 3.8, "SM配分にはSM数別測定曲線を利用")
bullet(s, 8.45, 5.15, 3.8, "最終性能はSTG計算値ではなく実GPUで測定")
text(s, 0.8, 6.35, 11.8, 0.35, "例：Deep Learningではconv2d(x, layer1)が長く、後続のx系列とconcat開始時刻を拘束する。", 15, BLACK, True, PP_ALIGN.CENTER)

# 6. Proposed method
s = slide("04 / 提案手法", "二段階ヒューリスティック")
boxes = [
    ("Stage 1", "重要度計算", "bottom levelで後続に長い経路を持つタスクを優先"),
    ("Stage 2", "Stream配置", "readyタスクを予測終了時刻が最小のStreamへ配置"),
    ("Stage 3", "SM配分", "配置を固定し、実測プロファイルで候補を探索"),
]
for i, (stage, title, detail) in enumerate(boxes):
    x = 0.75 + i * 4.15
    box(s, x, 2.05, 3.55, 2.55, LIGHT, MID)
    text(s, x + 0.2, 2.3, 3.1, 0.3, stage, 13, GRAY, True)
    text(s, x + 0.2, 2.75, 3.1, 0.45, title, 21, BLACK, True)
    text(s, x + 0.2, 3.45, 3.1, 0.7, detail, 14, DARK)
    if i < 2:
        line(s, x + 3.58, 3.3, x + 4.08, 3.3, BLACK, 1.5, True)
box(s, 1.15, 5.15, 11.0, 0.9, WHITE, BLACK)
text(s, 1.35, 5.38, 10.6, 0.35, "DAGの依存関係を守る  +  実カーネルのSM依存性を使う  →  実GPUで検証", 18, BLACK, True, PP_ALIGN.CENTER)

# 7. Experimental design
s = slide("05 / 実験設計", "比較対象と測定条件")
box(s, 0.75, 1.95, 5.7, 4.45, LIGHT, MID)
text(s, 1.05, 2.25, 4.8, 0.35, "比較する4方式", 19, BLACK, True)
methods = [
    "Sequential：依存順に逐次実行",
    "Existing：既存のStream配置",
    "Existing + GC：既存配置 + Green Context",
    "Proposed：重要度配置 + SM配分",
]
for i, value in enumerate(methods):
    bullet(s, 1.05, 2.9 + i * 0.62, 4.9, value, 15)
box(s, 6.8, 1.95, 5.75, 4.45, WHITE, BLACK)
text(s, 7.1, 2.25, 4.8, 0.35, "評価", 19, BLACK, True)
for i, value in enumerate([
    "合成STG：5種類",
    "実アプリケーション：6種類",
    "各条件33回、3回ウォームアップ",
    "残り30回の平均・95% CI",
    "全24条件で数値検証PASS",
]):
    bullet(s, 7.1, 2.9 + i * 0.62, 4.9, value, 15)

# 8. Synthetic results
s = slide("06 / 結果", "合成STG：グラフ構造により最良方式が変わる")
image(s, "paper/evidence_figures/synthetic_method_comparison.png", 0.55, 1.7, w=7.15)
box(s, 8.05, 1.95, 4.65, 3.95, LIGHT, MID)
text(s, 8.35, 2.25, 3.9, 0.35, "観察", 19, BLACK, True)
bullet(s, 8.35, 2.9, 3.9, "Mixed Chain + ParallelではProposedが短い")
bullet(s, 8.35, 3.7, 3.9, "Fully Parallel等ではExistingが短い")
bullet(s, 8.35, 4.5, 3.9, "単一方式の常勝ではない")
text(s, 0.8, 6.35, 11.8, 0.35, "比較結果は、手法の優劣を一つに決めるより、DAG構造と方式の適合条件を示す。", 16, BLACK, True, PP_ALIGN.CENTER)

# 9. Real application results
s = slide("06 / 結果", "実アプリケーション：提案方式は全ケースで最速ではない")
image(s, "paper/evidence_figures/real_application_four_methods.png", 0.55, 1.72, w=7.35)
box(s, 8.2, 1.95, 4.45, 3.95, LIGHT, MID)
text(s, 8.5, 2.25, 3.7, 0.35, "代表例", 19, BLACK, True)
bullet(s, 8.5, 2.9, 3.7, "Image Processing：Proposed 64.54 ms")
bullet(s, 8.5, 3.75, 3.7, "Deep Learning：Existing + GC 120.94 ms")
bullet(s, 8.5, 4.6, 3.7, "Proposedは136.01 ms")
text(s, 0.8, 6.35, 11.8, 0.35, "結論：提案方式の有効性は、入力DAGと実カーネルの特性に依存する。", 17, BLACK, True, PP_ALIGN.CENTER)

# 10. Ablation
s = slide("07 / アブレーション", "何が効いたのかを要素別に確認する")
image(s, "paper/results_report/figures/application_ablation_execution_time.png", 0.55, 1.72, w=7.25)
box(s, 8.1, 1.95, 4.55, 3.95, LIGHT, MID)
text(s, 8.4, 2.25, 3.8, 0.35, "Fullとの比較", 19, BLACK, True)
bullet(s, 8.4, 2.9, 3.8, "No Stream placementで大幅悪化")
bullet(s, 8.4, 3.7, 3.8, "Image：No SM allocationが僅かに短い")
bullet(s, 8.4, 4.5, 3.8, "Deep：No SM allocationが短い")
text(s, 0.8, 6.35, 11.8, 0.35, "Stream配置の寄与は確認できるが、重要度とSM配分の追加効果は一様ではない。", 16, BLACK, True, PP_ALIGN.CENTER)

# 11. DAG sensitivity
s = slide("08 / 構造分析", "Image ProcessingとDeep Learningで効果が異なる理由")
image(s, "paper/evidence_figures/image_deep_stream_sensitivity.png", 0.55, 1.78, w=6.9)
box(s, 7.8, 1.95, 4.85, 3.95, LIGHT, MID)
text(s, 8.1, 2.25, 4.1, 0.35, "DAG構造", 19, BLACK, True)
bullet(s, 8.1, 2.9, 4.1, "Image：複数分岐で並列化余地")
bullet(s, 8.1, 3.7, 4.1, "4 Stream付近で最短、5で悪化")
bullet(s, 8.1, 4.5, 4.1, "Deep：2本の長い分岐がconcatで合流")
bullet(s, 8.1, 5.25, 4.1, "遅い分岐と合流待ちが支配")

# 12. Conclusion
s = slide("09 / 結論", "言えること・言えないこと")
box(s, 0.75, 1.9, 5.8, 4.5, LIGHT, MID)
text(s, 1.05, 2.2, 5.0, 0.35, "言えること", 20, BLACK, True)
for i, value in enumerate([
    "複数Streamは多くの実アプリで有効",
    "Stream配置は独立分岐の多いDAGで効きやすい",
    "重要度・SM配分の効果はグラフ依存",
    "提案方式は条件付きで有効なヒューリスティック",
]):
    bullet(s, 1.05, 2.85 + i * 0.65, 5.0, value, 15)
box(s, 6.85, 1.9, 5.8, 4.5, WHITE, BLACK)
text(s, 7.15, 2.2, 5.0, 0.35, "言えないこと", 20, BLACK, True)
for i, value in enumerate([
    "すべてのDAGで最速",
    "SM使用率最大化 = makespan最小",
    "異なるGPU・入力でも一般化済み",
    "厳密最適解を求める方式",
]):
    bullet(s, 7.15, 2.85 + i * 0.65, 5.0, value, 15)

prs.save(OUT)
print(OUT)
