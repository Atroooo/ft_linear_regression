import pandas as pd
import json
import matplotlib.pyplot as plt
from ft_linear_regression import LinearRegression


def prepare_data(path):
    """Read the data from a csv file and
    return the values of the columns 'km' and 'price'.

    Args:
        path (str): path to the csv file.

    Returns:
        lists: values of the columns 'km' and 'price'.
    """
    try:
        data = pd.read_csv(path)
        X = data['km'].values
        y = data['price'].values
    except FileNotFoundError:
        print("File not found")
        exit(1)
    return X, y


def normalisation(s):
    """Normalise the data.

    Args:
        s (list): list of values to normalise.

    Returns:
        list: list of normalised values.
    """
    return [((x - min(s)) / (max(s) - min(s))) for x in s]


def save_params(X, y, w, b):
    """Scale back the parameters and save them in a json file.

    Args:
        X (list): Car mileage.
        y (list): Car price.
        w (float): weight.
        b (float): bias.
    """
    deltaX = max(X) - min(X)
    deltaY = max(y) - min(y)
    t0 = ((deltaY * b) + min(y) - w * (deltaY / deltaX) * min(X))
    t1 = deltaY * w / deltaX
    print(f"theta0 = {t0}, theta1 = {t1}")

    with open('params.json', 'w') as f:
        json.dump({'theta0': t0, 'theta1': t1}, f)


def plot_result(nX, ny, w, b):
    plt.yscale('linear')
    plt.scatter(nX, ny, color='red')
    plt.axline((0, b), slope=w)
    plt.show()


if __name__ == "__main__":
    X, y = prepare_data("data.csv")
    nX = normalisation(X)
    ny = normalisation(y)
    ln = LinearRegression(lr=0.01, epochs=5000)
    ln.train(nX, ny)
    w, b = ln.get_params()
    save_params(X, y, w, b)
    plot_result(nX, ny, w, b)
