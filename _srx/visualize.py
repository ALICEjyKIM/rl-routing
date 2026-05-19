import matplotlib.pyplot as plt

def plot_vrp(data):
    customers = data['customers']
    depots = data['depots']

    # 고객 좌표
    x_c = [c['x'] for c in customers]
    y_c = [c['y'] for c in customers]

    # depot 좌표
    x_d = [d['x'] for d in depots]
    y_d = [d['y'] for d in depots]

    # plot
    plt.figure(figsize=(6,6))
    plt.scatter(x_c, y_c, label="Customers")
    plt.scatter(x_d, y_d, marker='s', label="Depot")

    plt.legend()
    plt.title("VRP Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()