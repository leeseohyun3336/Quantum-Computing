"""
후보 경로 선택형 QUBO - 최종 실험 코드

같은 폴더에 grid_map.py, smoke_scenario.py 가 있어야 함
필요 라이브러리: py -m pip install dimod dwave-samplers

[무엇을 비교하나]
  기준선  : 로봇마다 "자기 혼자 기준" 최선 후보(출발 대기 없음)를 고르고, 출구에서는 도착 순서대로 통과
            (출구 통과 시간·폐쇄 시각까지 고려하지만, 다른 로봇은 고려하지 않음)
  QUBO    : 세 로봇 후보의 "조합"을 한 번에 고름 → 충돌·출구 병목을 서로 고려
            풀이: 전수 탐색(정답), Simulated Annealing(고전 어닐링), (선택) D-Wave QPU

[QUBO]
  변수 x[r,k] = 1 : 로봇 r이 후보 k 선택  (k = FAIL 이면 대피 포기)
  최소화  Σ 비용[r,k]·x[r,k]
        + P·Σ_r (Σ_k x[r,k] − 1)²                 로봇당 정확히 1개
        + P·x[r,k]·x[s,l]  (충돌 또는 병목)        두 후보를 같이 고르면 안 되는 경우
  비용 = 대피 완료 시각 + ALPHA × 누적 연기 위험,  FAIL 비용 = 100
"""

import itertools
import json
import time

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from grid_map import COLS, EXIT, OBSTACLE, ROBOT_STARTS, ROWS, build_grid, exits, neighbors
from smoke_scenario import ALPHA, BLOCK_LEVEL, EXIT_SERVICE, T_MAX, exit_close_time, make_smoke

# ---- 설정 ---------------------------------------------------------------------
MAX_DELAY = 4        # 출발 전 대기 0~4초 버전을 후보로 생성
FAIL_COST = 100.0    # 대피 포기 비용 (어떤 실제 경로보다 훨씬 크게)
PENALTY = 200.0      # 제약 페널티 (FAIL_COST 보다 크게)
USE_DWAVE = False    # D-Wave API 토큰이 있으면 True

# 시나리오: (발화점, 연기가 1칸 퍼지는 데 걸리는 초)
SCENARIOS = {
    "S1 E2근처-느림": ((8, 9), 4),
    "S2 E2근처-보통": ((8, 9), 2),
    "S3 E2근처-빠름": ((8, 9), 1),
    "N1 E2근처(8,8)-보통": ((8, 8), 2),
    "N2 북쪽설비(0,7)-빠름": ((0, 7), 1),
}
FAIL = "FAIL"


# ---- 1. 후보 경로 생성 (로봇 1대씩, 고전 계산) --------------------------------
def plan_to_exit(grid, smoke, start, exit_cell, delay):
    import heapq
    for t in range(delay + 1):
        if smoke[t, start[0], start[1]] >= BLOCK_LEVEL:
            return None
    pq, best, parent = [(0.0, delay, start)], {(start, delay): 0.0}, {}
    while pq:
        cost, t, cell = heapq.heappop(pq)
        if cell == exit_cell:
            path, key = [cell], (cell, t)
            while key in parent:
                key = parent[key]
                path.append(key[0])
            return [start] * delay + path[::-1]
        if t >= T_MAX or cost > best.get((cell, t), np.inf):
            continue
        for nxt in [cell, *neighbors(grid, cell)]:
            s = smoke[t + 1, nxt[0], nxt[1]]
            if s >= BLOCK_LEVEL:
                continue
            nc = cost + 1 + ALPHA * s
            if nc < best.get((nxt, t + 1), np.inf):
                best[(nxt, t + 1)] = nc
                parent[(nxt, t + 1)] = (cell, t)
                heapq.heappush(pq, (nc, t + 1, nxt))
    return None


def make_candidates(grid, smoke):
    exit_cells = exits(grid)
    cands = {}
    for name, start in ROBOT_STARTS.items():
        seen, lst = set(), []
        for ei, ex in enumerate(exit_cells):
            close = exit_close_time(smoke, ex)
            for d in range(MAX_DELAY + 1):
                p = plan_to_exit(grid, smoke, start, ex, d)
                if p is None or tuple(p) in seen:
                    continue
                arrive = len(p) - 1
                done = arrive + EXIT_SERVICE
                if close is not None and done > close:
                    continue
                seen.add(tuple(p))
                risk = sum(smoke[t, r, c] for t, (r, c) in enumerate(p))
                risk += sum(smoke[min(t, T_MAX), ex[0], ex[1]] for t in range(arrive, done))
                lst.append({"path": p, "exit": ei, "arrive": arrive, "done": done, "delay": d,
                            "risk": float(risk), "cost": done + ALPHA * float(risk)})
        cands[name] = lst
    return cands


# ---- 2. 두 후보를 같이 고를 수 없는지 (충돌·병목) ------------------------------
def conflict(a, b):
    pa, pb = a["path"], b["path"]
    for t in range(min(len(pa), len(pb))):
        both_arrived = t == len(pa) - 1 and t == len(pb) - 1
        if pa[t] == pb[t] and not both_arrived:
            return True
        if t > 0 and pa[t] == pb[t - 1] and pb[t] == pa[t - 1]:
            return True
    if a["exit"] == b["exit"] and a["arrive"] < b["done"] and b["arrive"] < a["done"]:
        return True
    return False


# ---- 3. 기준선 ----------------------------------------------------------------
def baseline(grid, smoke, cands):
    """로봇마다 혼자 기준 최선 후보(대기 없음) → 출구에서 도착 순서대로 통과"""
    exit_cells = exits(grid)
    pick = {}
    for r, lst in cands.items():
        own = [c for c in lst if c["delay"] == 0]
        pick[r] = min(own, key=lambda c: c["cost"]) if own else None
    out, by_exit = {}, {}
    for r, c in pick.items():
        if c is None:
            out[r] = None
        else:
            by_exit.setdefault(c["exit"], []).append((c["arrive"], r))
    for e, arrivals in by_exit.items():
        free_at, close = 0, exit_close_time(smoke, exit_cells[e])
        for arrive, r in sorted(arrivals):
            start = max(arrive, free_at)
            done = start + EXIT_SERVICE
            free_at = done
            c = pick[r]
            wait_risk = sum(smoke[min(t, T_MAX), exit_cells[e][0], exit_cells[e][1]]
                            for t in range(c["done"], done))
            ok = close is None or done <= close
            out[r] = {"exit": e, "done": done, "wait": start - arrive,
                      "risk": c["risk"] + float(wait_risk)} if ok else None
    return out, pick


# ---- 4. QUBO ------------------------------------------------------------------
def build_bqm(cands):
    bqm = dimod.BinaryQuadraticModel("BINARY")
    names = list(cands)
    for r in names:
        for k, c in enumerate(cands[r]):
            bqm.add_variable((r, k), c["cost"])
        bqm.add_variable((r, FAIL), FAIL_COST)
        bqm.add_linear_equality_constraint(
            [((r, k), 1.0) for k in range(len(cands[r]))] + [((r, FAIL), 1.0)],
            lagrange_multiplier=PENALTY, constant=-1.0)
    for i, r in enumerate(names):
        for s in names[i + 1:]:
            for k, a in enumerate(cands[r]):
                for l, b in enumerate(cands[s]):
                    if conflict(a, b):
                        bqm.add_interaction((r, k), (s, l), PENALTY)
    return bqm


def check(choice, cands):
    """선택이 유효한지: 로봇당 1개, 충돌·병목 없음"""
    if any(v is None for v in choice.values()):
        return False
    real = [(r, k) for r, k in choice.items() if k != FAIL]
    return not any(conflict(cands[a][ka], cands[b][kb])
                   for (a, ka), (b, kb) in itertools.combinations(real, 2))


def decode(sample, cands):
    choice = {}
    for r in cands:
        on = [k for k in [*range(len(cands[r])), FAIL] if sample[(r, k)] == 1]
        choice[r] = on[0] if len(on) == 1 else None
    return choice


def exact(cands):
    names, best, best_val = list(cands), None, np.inf
    for combo in itertools.product(*[[*range(len(cands[r])), FAIL] for r in names]):
        ch = dict(zip(names, combo))
        if not check(ch, cands):
            continue
        val = sum(FAIL_COST if k == FAIL else cands[r][k]["cost"] for r, k in ch.items())
        if val < best_val:
            best, best_val = ch, val
    return best


def summarize_choice(choice, cands):
    res = {}
    for r, k in choice.items():
        if k == FAIL:
            res[r] = None
        else:
            c = cands[r][k]
            res[r] = {"exit": c["exit"], "done": c["done"], "wait": c["delay"], "risk": c["risk"]}
    return res


def metrics(res):
    ok = [v for v in res.values() if v]
    return {
        "대피 성공": f"{len(ok)}/{len(res)}",
        "전체 대피(초)": max((v["done"] for v in ok), default=None),
        "대기 합(초)": sum(v["wait"] for v in ok),
        "연기 위험 합": round(sum(v["risk"] for v in ok), 2),
    }


# ---- 5. 그림 ------------------------------------------------------------------
def plot(grid, smoke, cands, base_pick, qubo_choice, title, path, base_failed=()):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    colors = {"R1": "#2b6cb0", "R2": "#c05621", "R3": "#6b46c1"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [("기준선 (각자 최선)", base_pick, set(base_failed)), ("QUBO (조합 최적)", qubo_choice, set())]
    for ax, (label, choice, failed) in zip(axes, panels):
        t_show = 12
        base = np.where(grid == OBSTACLE, 1, np.where(grid == EXIT, 2, 0))
        ax.imshow(base, cmap=ListedColormap(["#f4f4f2", "#5b6270", "#2e9d5b"]), vmin=0, vmax=2)
        s = np.ma.masked_where((smoke[t_show] == 0) | (grid == OBSTACLE), smoke[t_show])
        ax.imshow(s, cmap="Reds", vmin=0, vmax=1, alpha=0.6)
        for i, (er, ec) in enumerate(exits(grid)):
            ax.text(ec, er, f"E{i+1}", ha="center", va="center", color="white", weight="bold")
        for r, c in choice.items():
            if c is None:
                continue
            p = c["path"]
            off = {"R1": -0.12, "R2": 0.0, "R3": 0.12}[r]
            ax.plot([q[1] + off for q in p], [q[0] + off for q in p], color=colors[r], lw=2.2,
                    ls="--" if r in failed else "-")
            sr, sc = p[0]
            ax.add_patch(plt.Circle((sc, sr), 0.33, color=colors[r]))
            ax.text(sc, sr, r, ha="center", va="center", color="white", fontsize=7, weight="bold")
            if r in failed:
                ax.text(sc, sr + 0.42, "대피 실패", ha="center", va="top", color="#b00020", fontsize=8, weight="bold")
        ax.set_xticks(np.arange(-0.5, COLS), minor=True)
        ax.set_yticks(np.arange(-0.5, ROWS), minor=True)
        ax.grid(which="minor", color="#c9c9c9", lw=0.6)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(label)
    fig.suptitle(f"{title}  (붉은색: t={t_show}초 연기)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- 실행 ---------------------------------------------------------------------
if __name__ == "__main__":
    grid = build_grid()
    sampler = SimulatedAnnealingSampler()
    if USE_DWAVE:
        from dwave.system import DWaveSampler, EmbeddingComposite
        qpu = EmbeddingComposite(DWaveSampler())
    table = {}

    for label, (origin, interval) in SCENARIOS.items():
        smoke = make_smoke(grid, interval, origin=origin)
        cands = make_candidates(grid, smoke)
        bqm = build_bqm(cands)
        close = {f"E{i+1}": exit_close_time(smoke, e) for i, e in enumerate(exits(grid))}
        print(f"\n===== {label}  발화점 {origin}, 연기 1칸/{interval}초, 출구 폐쇄 {close} =====")
        print(f"QUBO 변수 {bqm.num_variables}개, 상호작용 {bqm.num_interactions}개")

        base_res, base_pick = baseline(grid, smoke, cands)

        t0 = time.time()
        ex = exact(cands)
        t_ex = time.time() - t0

        t0 = time.time()
        sa = decode(sampler.sample(bqm, num_reads=300, seed=1).first.sample, cands)
        t_sa = time.time() - t0
        sa_ok = check(sa, cands)

        rows = {"기준선": metrics(base_res), "QUBO-정답": metrics(summarize_choice(ex, cands))}
        rows["QUBO-정답"]["계산(초)"] = round(t_ex, 3)
        rows["QUBO-SA"] = metrics(summarize_choice(sa, cands)) if sa_ok else {"대피 성공": "제약 위반"}
        rows["QUBO-SA"]["계산(초)"] = round(t_sa, 3)

        if USE_DWAVE:
            t0 = time.time()
            qs = decode(qpu.sample(bqm, num_reads=300).first.sample, cands)
            t_q = time.time() - t0
            rows["QUBO-QPU"] = metrics(summarize_choice(qs, cands)) if check(qs, cands) else {"대피 성공": "제약 위반"}
            rows["QUBO-QPU"]["계산(초)"] = round(t_q, 3)

        for k, v in rows.items():
            print(f"  {k:10s} {v}")
        for r, k in ex.items():
            if k == FAIL:
                print(f"    {r}: 대피 포기")
            else:
                c = cands[r][k]
                print(f"    {r}: E{c['exit']+1}, 출발 대기 {c['delay']}초, 도착 {c['arrive']}초, 완료 {c['done']}초")
        table[label] = rows

        tag = label.split()[0]
        plot(grid, smoke, cands, base_pick,
             {r: (None if k == FAIL else cands[r][k]) for r, k in ex.items()},
             label, f"compare_{tag}.png", base_failed=[r for r, v in base_res.items() if v is None])

    with open("candidate_results.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print("\n그림 compare_*.png, 결과 candidate_results.json 저장 완료")
