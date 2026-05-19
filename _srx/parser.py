def parse_vrp(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # ------------------
    # 1. 첫 줄
    # ------------------
    type_, m, n, t = map(int, lines[0].split())

    idx = 1

    # ------------------
    # 2. depot 정보 (capacity)
    # ------------------
    depot_info = []
    for _ in range(t):
        D, Q = map(int, lines[idx].split())
        depot_info.append(Q)
        idx += 1

    # ------------------
    # 3. 고객 정보
    # ------------------
    customers = []
    for _ in range(n):
        parts = list(map(int, lines[idx].split()))

        customer = {
            "id": parts[0],
            "x": parts[1],
            "y": parts[2],
            "demand": parts[4]
        }

        customers.append(customer)
        idx += 1

    # ------------------
    # 4. depot 좌표
    # ------------------
    depots = []
    for _ in range(t):
        parts = list(map(int, lines[idx].split()))

        depot = {
            "id": parts[0],
            "x": parts[1],
            "y": parts[2]
        }

        depots.append(depot)
        idx += 1

    return {
        "m": m,
        "capacity": depot_info[0],  # 일단 동일하다고 가정
        "customers": customers,
        "depots": depots
    }