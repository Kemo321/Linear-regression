import matplotlib.pyplot as plt


class Plot:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot(x, y, pred, title):
        plt.plot(x, pred, label=title)
        plt.scatter(x, y, color='red', label='Original')
        plt.xlabel(title)
        plt.ylabel("Sales")
        plt.legend()
