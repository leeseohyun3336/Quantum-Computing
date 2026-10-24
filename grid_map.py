"""
공장 격자 지도 (12 x 10)

- 행(row) 10개 x 열(col) 12개, 좌표 표기는 (행, 열), 왼쪽 위가 (0, 0)
- 1칸 = 1.6m (Spot 최대 속도 1.6 m/s 기준, 1스텝 = 1초에 1칸 이동)
- 실제 크기: 가로 19.2m x 세로 16m

셀 값
  0 = 통로 (이동 가능)
  1 = 설비 / 벽 (이동 불가)
  2 = 비상구
"""

from collections import deque
import json

import numpy as np

ROWS, COLS = 10, 12
CELL_SIZE_M = 1.6
STEP_SEC = 1.0

PATH, OBSTACLE, EXIT = 0, 1, 2

# ---- 지도 (행 0~9, 열 0~11) -------------------------------------------
# . = 통로, # = 설비, E = 비상구
LAYOUT = [
    "............",  # 0
    "..###..###..",  # 1  설비 A, 설비 B
    "..###..###..",  # 2
    "............",  # 3  주 통로
    "E...........",  # 4  비상구 1 (서쪽)
    "..##..####..",  # 5  설비 C, 설비 D
    "..##..####..",  # 6
    "..##........",  # 7
    "............",  # 8
    "...........E",  # 9  비상구 2 (동남쪽)
]

# 로봇 시작 위치 (행, 열)
ROBOT_STARTS = {
    "R1": (0, 5),
    "R2": (3, 10),
    "R3": (8, 4),
}


def build_grid():
    symbol = {".": PATH, "#": OBSTACLE, "E": EXIT}
    grid = np.array([[symbol[c] for c in row] for row in LAYOUT], dtype=int)
    assert grid.shape == (ROWS, COLS), grid.shape
    return grid


def exits(grid):
    return [tuple(map(int, p)) for p in np.argwhere(grid == EXIT)]


def neighbors(grid, cell):
    """상하좌우 4방향 이동 (대각선 없음)"""
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr, nc] != OBSTACLE:
            yield (nr, nc)


def bfs_distance(grid, start):
    """start에서 모든 칸까지의 최단 칸 수 (연기 없는 상태)"""
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in neighbors(grid, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def check(grid):
    """모든 통로 칸에서 두 비상구 모두 도달 가능한지, 로봇 위치가 통로인지 확인"""
    free = {tuple(map(int, p)) for p in np.argwhere(grid != OBSTACLE)}
    for name, pos in ROBOT_STARTS.items():
        assert grid[pos] == PATH, f"{name} 시작 위치가 통로가 아님: {pos}"
    for e in exits(grid):
        reach = bfs_distance(grid, e)
        assert set(reach) == free, f"비상구 {e}에서 도달 못하는 칸이 있음"
    print("지도 확인 완료: 모든 통로에서 두 비상구 모두 도달 가능")
    print(f"통로 {int((grid == PATH).sum())}칸 / 설비 {int((grid == OBSTACLE).sum())}칸 / 비상구 {len(exits(grid))}개")
    print("\n로봇별 각 비상구까지 최단 거리 (연기 없음 기준)")
    for name, pos in ROBOT_STARTS.items():
        d = bfs_distance(grid, pos)
        txt = ", ".join(f"비상구{i+1}{e}: {d[e]}칸({d[e]*CELL_SIZE_M:.1f}m, {d[e]*STEP_SEC:.0f}초)"
                        for i, e in enumerate(exits(grid)))
        print(f"  {name} {pos} -> {txt}")


def save(grid, prefix="grid"):
    np.savetxt(f"{prefix}.csv", grid, fmt="%d", delimiter=",")
    config = {
        "rows": ROWS,
        "cols": COLS,
        "cell_size_m": CELL_SIZE_M,
        "step_sec": STEP_SEC,
        "cell_values": {"0": "통로", "1": "설비/벽", "2": "비상구"},
        "coordinate": "(행, 열), 왼쪽 위 (0,0)",
        "move": "상하좌우 4방향, 스텝당 1칸",
        "exits": exits(grid),
        "robot_starts": ROBOT_STARTS,
        "grid": grid.tolist(),
    }
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def plot(grid, path="grid.png"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # 한글 폰트: 설치된 것 중 먼저 찾은 것을 사용 (Windows는 맑은 고딕)
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

    cmap = ListedColormap(["#f4f4f2", "#5b6270", "#2e9d5b"])
    fig, ax = plt.subplots(figsize=(8, 6.8))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticks(np.arange(-0.5, COLS), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS), minor=True)
    ax.grid(which="minor", color="#c9c9c9", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("열 (col)")
    ax.set_ylabel("행 (row)")

    labels = {(1.5, 3): "설비 A", (1.5, 8): "설비 B", (6, 2.5): "설비 C", (5.5, 7.5): "설비 D"}
    for (r, c), t in labels.items():
        ax.text(c, r, t, ha="center", va="center", color="white", fontsize=10)

    for i, (r, c) in enumerate(exits(grid)):
        ax.text(c, r, f"E{i+1}", ha="center", va="center", color="white", fontsize=11, weight="bold")

    for name, (r, c) in ROBOT_STARTS.items():
        ax.add_patch(plt.Circle((c, r), 0.35, color="#2b6cb0"))
        ax.text(c, r, name, ha="center", va="center", color="white", fontsize=9, weight="bold")

    ax.set_title(f"공장 격자 지도 {COLS}×{ROWS}  (1칸 = {CELL_SIZE_M}m, 전체 {COLS*CELL_SIZE_M:.1f}m × {ROWS*CELL_SIZE_M:.1f}m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()  # 창으로 띄우기


if __name__ == "__main__":
    g = build_grid()
    print(g)
    check(g)
    save(g)
    plot(g)