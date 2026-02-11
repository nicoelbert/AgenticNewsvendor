# simulator/utils.py
import matplotlib.pyplot as plt


def plot_feature(df, col):
    plt.figure(figsize=(10, 3))
    plt.plot(df.index, df[col])
    plt.title(f"Feature: {col}")
    plt.show()


def plot_demand(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["demand"], label="Demand")
    plt.title("Simulated demand")
    plt.legend()
    plt.show()
