import pandas as pd
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
from Plot import Plot


def main():
    data = pd.read_csv("Advertising.csv")
    tv = data[["TV", "sales"]]
    radio = data[["radio", "sales"]]
    newspaper = data[["newspaper", "sales"]]

    model = LinearRegression()

    model.fit(tv["TV"], tv["sales"])
    first = model.predict(tv["TV"])

    model.fit(radio["radio"], radio["sales"])
    second = model.predict(radio["radio"])

    model.fit(newspaper["newspaper"], newspaper["sales"])
    third = model.predict(newspaper["newspaper"])

    plt.figure(figsize=(10, 12))
    Plot.plot(tv["TV"], tv["sales"], first, "TV")
    plt.savefig("RSS/TV.png")
    plt.clf()
    Plot.plot(radio["radio"], radio["sales"], second, "Radio")
    plt.savefig("RSS/Radio.png")
    plt.clf()
    Plot.plot(newspaper["newspaper"], newspaper["sales"], third, "Newspaper")
    plt.savefig("RSS/Newspaper.png")


if __name__ == "__main__":
    main()
