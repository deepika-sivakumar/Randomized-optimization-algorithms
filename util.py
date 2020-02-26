import matplotlib.pyplot as plt

def generate_graph(filename, title, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend()
    # plt.show()
    plt.savefig('graphs/'+ filename)
    plt.close()