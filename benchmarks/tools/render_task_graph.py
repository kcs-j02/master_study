from collections import defaultdict, deque
from html import escape


def load_stg(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    tasks = {}

    for line in lines[1:]:
        parts = list(map(int, line.split()))

        task_id = parts[0]
        proc_time = parts[1]
        num_pred = parts[2]
        preds = parts[3:3 + num_pred]

        tasks[task_id] = {
            "proc_time": proc_time,
            "preds": preds
        }

    print(f"tasks: {len(tasks)} / {n}")

    return tasks


def find_longest_path(tasks):
    """DAGの最長パスを求める"""

    successors = defaultdict(list)
    indegree = {tid: 0 for tid in tasks}

    for tid, task in tasks.items():
        for pred in task["preds"]:
            successors[pred].append(tid)
            indegree[tid] += 1

    # トポロジカルソート
    q = deque(
        tid for tid in tasks
        if indegree[tid] == 0
    )

    topo = []

    while q:
        u = q.popleft()
        topo.append(u)

        for v in successors[u]:
            indegree[v] -= 1

            if indegree[v] == 0:
                q.append(v)

    # 最長距離
    dist = {tid: 0 for tid in tasks}
    parent = {tid: None for tid in tasks}

    for u in topo:
        for v in successors[u]:
            if dist[v] < dist[u] + 1:
                dist[v] = dist[u] + 1
                parent[v] = u

    # 最長パス終端
    end = max(tasks, key=lambda x: dist[x])

    path = []

    while end is not None:
        path.append(end)
        end = parent[end]

    path.reverse()

    return path


def stg_to_svg(input_file, output_file):
    tasks = load_stg(input_file)

    # -------------------------
    # メインの長い直列経路
    # -------------------------
    main_path = find_longest_path(tasks)
    main_set = set(main_path)

    print(f"main path length: {len(main_path)}")

    # それ以外の分岐タスク
    branch_nodes = [
        tid for tid in sorted(tasks)
        if tid not in main_set
    ]

    # -------------------------
    # 描画設定
    # -------------------------
    margin = 40

    node_w = 46
    node_h = 18

    main_x = 80
    y_step = 25

    branch_start_x = 200

    branch_cols = 10
    branch_x_step = 65

    # SVG高さ
    height = (
        margin * 2
        + len(main_path) * y_step
    )

    width = (
        branch_start_x
        + branch_cols * branch_x_step
        + 100
    )

    positions = {}

    # -------------------------
    # メインパス配置
    # -------------------------
    for i, tid in enumerate(main_path):

        x = main_x
        y = margin + i * y_step

        positions[tid] = (x, y)

    # -------------------------
    # 分岐タスク配置
    # -------------------------
    if branch_nodes:

        rows = (
            len(branch_nodes)
            + branch_cols - 1
        ) // branch_cols

        top_y = margin + 40
        bottom_y = height - margin - 40

        branch_y_step = (
            (bottom_y - top_y)
            / max(rows - 1, 1)
        )

        for i, tid in enumerate(branch_nodes):

            row = i // branch_cols
            col = i % branch_cols

            x = branch_start_x + col * branch_x_step
            y = top_y + row * branch_y_step

            positions[tid] = (x, y)

    # -------------------------
    # メインパスの辺
    # -------------------------
    main_edges = set()

    for i in range(len(main_path) - 1):
        main_edges.add(
            (main_path[i], main_path[i + 1])
        )

    # -------------------------
    # SVG生成
    # -------------------------
    svg = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )

    svg.append("""
<defs>
    <marker id="arrow"
            markerWidth="6"
            markerHeight="6"
            refX="5"
            refY="3"
            orient="auto">
        <path d="M0,0 L0,6 L6,3 z"
              fill="#666"/>
    </marker>
</defs>
""")

    svg.append(
        '<rect width="100%" height="100%" fill="white"/>'
    )

    # -------------------------
    # 辺を描画
    # -------------------------
    for tid, task in tasks.items():

        x2, y2 = positions[tid]

        for pred in task["preds"]:

            if pred not in positions:
                continue

            x1, y1 = positions[pred]

            start_x = x1 + node_w / 2
            start_y = y1 + node_h

            end_x = x2 + node_w / 2
            end_y = y2

            if (pred, tid) in main_edges:
                stroke = "#222"
                opacity = "0.9"
                stroke_width = "1.4"
            else:
                stroke = "#888"
                opacity = "0.30"
                stroke_width = "0.7"

            svg.append(
                f'<line '
                f'x1="{start_x}" '
                f'y1="{start_y}" '
                f'x2="{end_x}" '
                f'y2="{end_y}" '
                f'stroke="{stroke}" '
                f'stroke-width="{stroke_width}" '
                f'opacity="{opacity}" '
                f'marker-end="url(#arrow)" />'
            )

    # -------------------------
    # ノードを描画
    # -------------------------
    for tid in sorted(tasks):

        x, y = positions[tid]

        if tid in main_set:
            stroke = "#111"
            stroke_width = "1.2"
        else:
            stroke = "#666"
            stroke_width = "0.8"

        proc_time = tasks[tid]["proc_time"]

        svg.append(
            f'<g>'
            f'<title>'
            f'Task {tid}, proc_time={proc_time}'
            f'</title>'
        )

        svg.append(
            f'<rect '
            f'x="{x}" '
            f'y="{y}" '
            f'width="{node_w}" '
            f'height="{node_h}" '
            f'rx="2" '
            f'fill="white" '
            f'stroke="{stroke}" '
            f'stroke-width="{stroke_width}" />'
        )

        svg.append(
            f'<text '
            f'x="{x + node_w / 2}" '
            f'y="{y + node_h / 2 + 3}" '
            f'text-anchor="middle" '
            f'font-family="Arial" '
            f'font-size="8">'
            f'{escape(str(tid))}'
            f'</text>'
        )

        svg.append('</g>')

    svg.append('</svg>')

    # -------------------------
    # 保存
    # -------------------------
    with open(output_file, "w") as f:
        f.write("\n".join(svg))

    print(f"saved: {output_file}")


if __name__ == "__main__":

    stg_to_svg(
        "common_sample.stg",
        "common_sample.svg"
    )