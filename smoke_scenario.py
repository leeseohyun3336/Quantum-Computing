"""
연기 확산 + 비상구 병목 시나리오 (기준선: 로봇별 독립 경로 탐색)

grid_map.py 와 같은 폴더에 두고 실행

하는 일
  1. 발화점에서 시간에 따라 퍼지는 연기 농도 smoke[t, 행, 열] (0~1) 생성
  2. 로봇마다 다른 로봇은 무시하고, 연기를 피해 비용이 가장 작은 출구·경로 선택
     (시간에 따라 연기가 변하므로 (칸, 시간) 상태로 Dijkstra)
  3. 출구 통과 시간(방화문)을 반영해 대기열 계산
  4. 지표 출력: 대피 완료 시간, 누적 연기 비용, 출구 대기 시간, 로봇 간 충돌 수
  5. 시나리오 3종(느림/보통/빠름) 비교 + 그림 저장

※ 이 방식은 로봇끼리 서로를 고려하지 않는 "기준선"이다.
   충돌·대기를 같이 고려하는 방법(우선순위 A*, QUBO)과 비교하는 출발점으로 쓴다.
"""

import heapq
import json

import numpy as np

from grid_map import (CELL_SIZE_M, COLS, EXIT, OBSTACLE, ROBOT_STARTS, ROWS,
                      bfs_distance, build_grid, exits, neighbors)

# ---- 파라미터 (가정값, 민감도 분석 대상) ----------------------------------
T_MAX = 60              # 시뮬레이션 최대 시간 (스텝 = 초)
FIRE_ORIGIN = (8, 9)    # 발화점 (E2 근처)
RAMP = 4                # 연기가 도착한 뒤 농도 1.0이 될 때까지 걸리는 스텝
BLOCK_LEVEL = 0.7       # 이 농도 이상이면 통행 불가 (센서·온도 한계 가정)
EXIT_SERVICE = 3        # 출구(방화문) 1대 통과에 걸리는 스텝
ALPHA = 5.0             # 연기 비용 가중치: 1스텝 비용 = 1 + ALPHA * 농도

SCENARIOS = {           # 연기 전선이 1칸 퍼지는 데 걸리는 스텝
    "S1 느림": 4,
    "S2 보통": 2,
    "S3 빠름": 1,
}


# ---- 1. 연기 확산 ----------------------------------------------------------
def make_smoke(grid, spread_interval, origin=FIRE_ORIGIN, t_max=T_MAX):
    """발화점에서 통로를 따라 퍼지는 연기.
    칸까지의 거리 d 만큼 떨어진 곳에 d * spread_interval 스텝에 도착하고,
    도착 후 RAMP 스텝 동안 0 -> 1 로 진해진다."""
    dist = bfs_distance(grid, origin)
    smoke = np.zeros((t_max + 1, ROWS, COLS))
    for (r, c), d in dist.items():
        arrive = d * spread_interval
        for t in range(arrive, t_max + 1):
            smoke[t, r, c] = min(1.0, (t - arrive + 1) / RAMP)
    return smoke


def exit_close_time(smoke, e):
    """출구가 연기로 사용 불가가 되는 시각 (없으면 None)"""
    idx = np.where(smoke[:, e[0], e[1]] >= BLOCK_LEVEL)[0]
    return int(idx[0]) if len(idx) else None


# ---- 2. 시간 확장 Dijkstra (로봇 1대) --------------------------------------
def plan_one(grid, smoke, start):
    """(칸, 시간) 상태에서 출구까지 비용 최소 경로. 대기(제자리)도 허용.
    반환: (비용, 경로 [(행,열), ...], 도착 출구)"""
    exit_set = set(exits(grid))
    pq = [(0.0, 0, start)]
    best = {(start, 0): 0.0}
    parent = {}
    while pq:
        cost, t, cell = heapq.heappop(pq)
        if cell in exit_set:
            path = [cell]
            key = (cell, t)
            while key in parent:
                key = parent[key]
                path.append(key[0])
            return cost, path[::-1], cell
        if t >= T_MAX or cost > best.get((cell, t), np.inf):
            continue
        for nxt in [cell, *neighbors(grid, cell)]:
            s = smoke[t + 1, nxt[0], nxt[1]]
            if s >= BLOCK_LEVEL:
                continue
            ncost = cost + 1 + ALPHA * s
            key = (nxt, t + 1)
            if ncost < best.get(key, np.inf):
                best[key] = ncost
                parent[key] = (cell, t)
                heapq.heappush(pq, (ncost, t + 1, nxt))
    return None, None, None


# ---- 3. 출구 대기열 + 지표 -------------------------------------------------
def evaluate(grid, smoke, plans):
    exit_names = {e: f"E{i+1}" for i, e in enumerate(exits(grid))}
    result = {}
    # 출구별 도착 순서대로 처리 (먼저 온 로봇부터, 1대씩 EXIT_SERVICE 스텝)
    by_exit = {}
    for name, (cost, path, ex) in plans.items():
        if path is None:
            result[name] = {"성공": False}
            continue
        by_exit.setdefault(ex, []).append((len(path) - 1, name))
    for ex, arrivals in by_exit.items():
        free_at = 0
        close = exit_close_time(smoke, ex)
        for arrive, name in sorted(arrivals):
            start = max(arrive, free_at)
            done = start + EXIT_SERVICE
            free_at = done
            cost, path, _ = plans[name]
            move_smoke = sum(smoke[t, r, c] for t, (r, c) in enumerate(path))
            wait_smoke = sum(smoke[min(t, T_MAX), ex[0], ex[1]] for t in range(arrive, done))
            ok = close is None or done <= close
            result[name] = {
                "성공": ok,
                "출구": exit_names[ex],
                "도착(초)": arrive,
                "대기(초)": start - arrive,
                "대피완료(초)": done,
                "누적연기": round(float(move_smoke + wait_smoke), 2),
            }
    return result


def count_conflicts(plans):
    """같은 시간 같은 칸(정점 충돌) + 서로 자리 바꾸기(간선 충돌). 출구 칸은 대기열이라 제외"""
    paths = {n: p for n, (_, p, _) in plans.items() if p}
    horizon = max(len(p) for p in paths.values())
    pos = lambda p, t: p[min(t, len(p) - 1)]
    names = list(paths)
    n = 0
    for t in range(horizon):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = paths[names[i]], paths[names[j]]
                if t >= len(a) - 1 and t >= len(b) - 1:
                    continue  # 둘 다 출구 도착 후
                if pos(a, t) == pos(b, t) and t < min(len(a), len(b)) - 1:
                    n += 1
                elif t > 0 and pos(a, t) == pos(b, t - 1) and pos(b, t) == pos(a, t - 1):
                    n += 1
    return n


def run(grid, spread_interval):
    smoke = make_smoke(grid, spread_interval)
    plans = {n: plan_one(grid, smoke, s) for n, s in ROBOT_STARTS.items()}
    res = evaluate(grid, smoke, plans)
    ok = [v for v in res.values() if v.get("성공")]
    summary = {
        "전체 대피 완료(초)": max(v["대피완료(초)"] for v in ok) if ok else None,
        "누적 연기 비용 합": round(sum(v["누적연기"] for v in ok), 2),
        "출구 대기 합(초)": sum(v["대기(초)"] for v in ok),
        "충돌 수": count_conflicts(plans),
        "대피 실패": [n for n, v in res.items() if not v.get("성공")],
        "출구 폐쇄 시각": {f"E{i+1}": exit_close_time(smoke, e) for i, e in enumerate(exits(grid))},
    }
    return smoke, plans, res, summary


# ---- 4. 시각화 -------------------------------------------------------------
def plot(grid, smoke, plans, title, path):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

    times = [0, 5, 10, 15]
    colors = {"R1": "#2b6cb0", "R2": "#c05621", "R3": "#6b46c1"}
    fig, axes = plt.subplots(1, len(times), figsize=(4.2 * len(times), 4.2))
    for ax, t in zip(axes, times):
        base = np.where(grid == OBSTACLE, 1, np.where(grid == EXIT, 2, 0))
        ax.imshow(base, cmap=ListedColormap(["#f4f4f2", "#5b6270", "#2e9d5b"]), vmin=0, vmax=2)
        s = np.ma.masked_where((smoke[t] == 0) | (grid == OBSTACLE), smoke[t])
        ax.imshow(s, cmap="Reds", vmin=0, vmax=1, alpha=0.75)
        r, c = FIRE_ORIGIN
        ax.text(c, r, "F", ha="center", va="center", color="black", weight="bold")
        for i, (er, ec) in enumerate(exits(grid)):
            ax.text(ec, er, f"E{i+1}", ha="center", va="center", color="white", weight="bold", fontsize=9)
        for name, (_, p, _) in plans.items():
            if not p:
                continue
            xs = [q[1] for q in p[: t + 1]]
            ys = [q[0] for q in p[: t + 1]]
            ax.plot(xs, ys, color=colors[name], linewidth=2)
            cr, cc = p[min(t, len(p) - 1)]
            ax.add_patch(plt.Circle((cc, cr), 0.33, color=colors[name]))
            ax.text(cc, cr, name, ha="center", va="center", color="white", fontsize=7, weight="bold")
        ax.set_xticks(np.arange(-0.5, COLS), minor=True)
        ax.set_yticks(np.arange(-0.5, ROWS), minor=True)
        ax.grid(which="minor", color="#c9c9c9", linewidth=0.6)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(f"t = {t}초")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


if __name__ == "__main__":
    grid = build_grid()
    print(f"발화점 {FIRE_ORIGIN}, 출구 통과 {EXIT_SERVICE}초/대, 통행 불가 농도 {BLOCK_LEVEL}\n")
    all_summary = {}
    for label, interval in SCENARIOS.items():
        smoke, plans, res, summary = run(grid, interval)
        all_summary[label] = summary
        print(f"=== {label} (연기 1칸 / {interval}초, 약 {CELL_SIZE_M / interval:.1f} m/s) ===")
        print(f"  출구 폐쇄 시각: {summary['출구 폐쇄 시각']}")
        for name, v in res.items():
            print(f"  {name}: {v}")
        print(f"  -> 전체 대피 {summary['전체 대피 완료(초)']}초, 대기 합 {summary['출구 대기 합(초)']}초, "
              f"연기 비용 {summary['누적 연기 비용 합']}, 충돌 {summary['충돌 수']}건, 실패 {summary['대피 실패']}\n")
        tag = label.split()[0]
        np.save(f"smoke_{tag}.npy", smoke)  # 다른 팀원이 불러 쓸 연기 데이터 smoke[t, 행, 열]
        plot(grid, smoke, plans, f"{label}: 연기 확산과 로봇별 독립 경로", f"scenario_{tag}.png")

    with open("scenario_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    import matplotlib.pyplot as plt
    plt.show()