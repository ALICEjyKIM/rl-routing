import math
import gurobipy as gp
from gurobipy import GRB


def euclidean_distance(node1, node2):
    dx = node1["x"] - node2["x"]
    dy = node1["y"] - node2["y"]
    return math.sqrt(dx * dx + dy * dy)


def solve_vrp_milp(data, time_limit=3600):
    """
    Multi-depot baseline MILP
    - vehicle k는 depot k에서 출발/복귀
    - p01처럼 m = t 인 경우를 기준으로 단순화
    """

    customers = data["customers"]
    depots = data["depots"]
    Q = data["capacity"]
    m = data["m"]

    customer_ids = [c["id"] for c in customers]
    depot_ids = [d["id"] for d in depots]
    vehicles = range(m)

    # depot 정보
    node_info = {}
    for d in depots:
        node_info[d["id"]] = {
            "x": d["x"],
            "y": d["y"],
            "demand": 0
        }

    for c in customers:
        node_info[c["id"]] = {
            "x": c["x"],
            "y": c["y"],
            "demand": c["demand"]
        }

    all_nodes = depot_ids + customer_ids

    # 차량 k의 home depot 지정
    # p01에서는 m=t=4라서 k번째 차량 ↔ k번째 depot
    vehicle_home_depot = {}
    for k in vehicles:
        vehicle_home_depot[k] = depot_ids[k % len(depot_ids)]

    # 거리 행렬
    dist = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                dist[i, j] = euclidean_distance(node_info[i], node_info[j])

    model = gp.Model("multi_depot_vrp")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPFocus", 1)      # feasible 먼저
    model.setParam("Heuristics", 0.5)

    # -----------------------------
    # 허용 arc 만들기
    # 각 차량은 자기 depot에서만 출발/복귀
    # depot-to-depot 이동은 금지
    # 다른 depot를 들르는 것도 금지
    # -----------------------------
    arcs = []

    for k in vehicles:
        home = vehicle_home_depot[k]

        # home depot -> customer
        for j in customer_ids:
            arcs.append((home, j, k))

        # customer -> home depot
        for i in customer_ids:
            arcs.append((i, home, k))

        # customer -> customer
        for i in customer_ids:
            for j in customer_ids:
                if i != j:
                    arcs.append((i, j, k))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    # MTZ 보조변수
    u = model.addVars(
        customer_ids,
        lb=0.0,
        ub=Q,
        vtype=GRB.CONTINUOUS,
        name="u"
    )

    # 목적함수
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j, k] for (i, j, k) in arcs),
        GRB.MINIMIZE
    )

    # -----------------------------
    # 제약식
    # -----------------------------

    # (1) 각 고객은 정확히 한 번 방문
    for j in customer_ids:
        model.addConstr(
            gp.quicksum(
                x[i, j, k]
                for (i, jj, k) in arcs
                if jj == j
            ) == 1,
            name=f"visit_once_{j}"
        )

    # (2) 각 고객은 정확히 한 번 떠남
    for i in customer_ids:
        model.addConstr(
            gp.quicksum(
                x[i, j, k]
                for (ii, j, k) in arcs
                if ii == i
            ) == 1,
            name=f"leave_once_{i}"
        )

    # (3) 각 차량은 자기 depot에서 최대 한 번 출발
    for k in vehicles:
        home = vehicle_home_depot[k]
        model.addConstr(
            gp.quicksum(
                x[home, j, k]
                for j in customer_ids
                if (home, j, k) in x
            ) <= 1,
            name=f"depart_home_{k}"
        )

    # (4) 각 차량은 자기 depot로 최대 한 번 복귀
    for k in vehicles:
        home = vehicle_home_depot[k]
        model.addConstr(
            gp.quicksum(
                x[i, home, k]
                for i in customer_ids
                if (i, home, k) in x
            ) <= 1,
            name=f"return_home_{k}"
        )

    # (5) 차량별 flow conservation
    for k in vehicles:
        for h in customer_ids:
            incoming = gp.quicksum(
                x[i, h, k]
                for i in all_nodes
                if i != h and (i, h, k) in x
            )
            outgoing = gp.quicksum(
                x[h, j, k]
                for j in all_nodes
                if j != h and (h, j, k) in x
            )
            model.addConstr(
                incoming == outgoing,
                name=f"flow_{h}_{k}"
            )

    # (6) MTZ + capacity
    demand = {c["id"]: c["demand"] for c in customers}

    for i in customer_ids:
        model.addConstr(u[i] >= demand[i], name=f"u_lb_{i}")
        model.addConstr(u[i] <= Q, name=f"u_ub_{i}")

    for k in vehicles:
        for i in customer_ids:
            for j in customer_ids:
                if i != j and (i, j, k) in x:
                    model.addConstr(
                        u[i] - u[j] + Q * x[i, j, k] <= Q - demand[j],
                        name=f"mtz_{i}_{j}_{k}"
                    )

    # 최적화
    model.optimize()

    


    print("=== SOLVER STATUS ===")
    print("Status code:", model.Status)

    if model.Status == GRB.OPTIMAL:
        print("최적해 찾음")
    elif model.Status == GRB.INFEASIBLE:
        print("❌ 모델 자체가 infeasible")
    elif model.Status == GRB.TIME_LIMIT:
        print("⏰ 시간 제한으로 중단")
    elif model.Status == GRB.INF_OR_UNBD:
        print("⚠ infeasible 또는 unbounded 가능")
    else:
        print("기타 상태:", model.Status)

    if model.SolCount == 0:
        return None, None

    # -----------------------------
    # 해 추출
    # -----------------------------
    routes = []

    for k in vehicles:
        home = vehicle_home_depot[k]

        starts = [
            j for j in customer_ids
            if (home, j, k) in x and x[home, j, k].X > 0.5
        ]

        if len(starts) == 0:
            continue

        route = [home]
        current = home
        visited = set()

        while True:
            next_node = None

            for j in all_nodes:
                if j != current and (current, j, k) in x and x[current, j, k].X > 0.5:
                    next_node = j
                    break

            if next_node is None:
                break

            route.append(next_node)

            if next_node == home:
                break

            if next_node in visited:
                break

            visited.add(next_node)
            current = next_node

        routes.append(route)

    return routes, model.ObjVal
