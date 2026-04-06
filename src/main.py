from parser import parse_vrp
from visualize import plot_vrp
from solver_milp import solve_vrp_milp

def main():
    data = parse_vrp("../data/p01_copy_2.txt")

    print("차량 수:", data["m"])
    print("capacity:", data["capacity"])
    print("고객 수:", len(data["customers"]))
    print("depot 수:", len(data["depots"]))

    #print(data)
    plot_vrp(data)

    total_demand = sum(c["demand"] for c in data["customers"])
    print("total demand:", total_demand)
    print("total capacity:", data["m"] * data["capacity"])
    
    routes, cost = solve_vrp_milp(data, time_limit=3600)

    print("routes:", routes)
    print("cost:", cost)
    

if __name__ == "__main__":
    main()