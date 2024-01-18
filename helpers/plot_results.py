import matplotlib.pyplot as plt


def plot_results(real, predicted, title="", save=False):
    plt.plot(real, label="Real values"),
    plt.plot(predicted, label="Predicted values")
    plt.ylabel("CO2 concentration")
    plt.xlabel("Iteration no.")
    plt.tight_layout()
    plt.legend()
    plt.title(title)
    fig1 = plt.gcf()
    plt.tight_layout()
    plt.show()
    if save:
        fig1.savefig("media/" + title + ".png", dpi=300)


def plot_colormap(y_test, y_pred, title="", save=False):
    diff = []
    for i in range(len(y_test)):
        diff.append(abs(y_test[i] - y_pred[i]))
    plt.scatter(y_test, y_pred, s=2, c=diff, cmap="cividis", )
    lim = [min(min(y_pred), min(y_test)), max(max(y_pred), max(y_test))]
    plt.plot(lim, lim, color="r")
    plt.xlabel("Real values")
    plt.ylabel("Predicted values")
    plt.title(title)
    plt.colorbar()
    # plt.clim(0, 150000)
    # plt.xlim(0, 150000)
    # plt.ylim(0, 150000)
    fig1 = plt.gcf()
    plt.tight_layout()
    plt.show()
    if save:
        fig1.savefig("media/" + title + ".png", dpi=300)