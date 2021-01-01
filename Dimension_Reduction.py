from matplotlib import pyplot as plt

def plot_PCA_2D(X_PCA, y_test):
    """
    Plots a 2D PCA plot
    :param X_PCA: X_Test after PCA fit.
    :param y_test: y_test
    :return: 2D PCA plot (green=Diagnosed, blue=Healthy)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_PCA[y_test == 0, 0], X_PCA[y_test == 0, 1], color='b')
    ax.scatter(X_PCA[y_test == 1, 0], X_PCA[y_test == 1, 1], color='g')
    ax.legend(('Healthy', 'Diagnosed'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1)
    ax.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1)
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('2D PCA')
    plt.show()