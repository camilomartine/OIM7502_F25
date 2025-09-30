import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # optional may be helpful for plotting percentage
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sb # optional to set plot theme
import yfinance as yf

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())


class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """method that downloads data and stores in a DataFrame
           uncomment the code below which should be the final two lines
           of your method"""
        data = self.calc_returns(yf.download(self.symbol, self.start, self.end, auto_adjust=False))
        return data

    def calc_returns(self, df):
        """method that adds change and return columns to data"""
        df['Change'] = df['Close'].pct_change()
        df['Instant_Return'] = np.log(df['Close']).diff().round(4)
        return df.dropna()

    def plot_return_dist(self):
        """method that plots instantaneous returns as histogram"""
        data_to_plot = self.data['Instant_Return'].values
        plt.hist(data_to_plot, bins=35, density=True, edgecolor='w', label='Instantaneous Returns')
        mean, std = norm.fit(data_to_plot)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = norm.pdf(x, mean, std)
        plt.plot(x, pdf, '--', linewidth=2, label=f'Normal Fit ($\mu$={mean:.4f}, $\sigma$={std:.4f})')

        # Formatting
        plt.title(f"{self.symbol} Instantaneous Return Distribution vs. Normal Distribution")
        plt.xlabel("Instantaneous Log Returns")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.show()

    def plot_performance(self):
        """method that plots stock object performance as percent """
        cum_log = self.data['Instant_Return'].cumsum()
        perf = np.exp(cum_log) * 100
        perf.index = self.data.index  # align index

        perf.plot()
        plt.title(f"{self.symbol} Performance Indexed to 100")
        plt.ylabel("Performance (%)")
        plt.xlabel("Date")
        plt.show()


def main():
    # uncomment (remove pass) code below to test
    test = Stock(symbol=['PLTR'])  # optionally test custom data range
    print(test.data)
    test.plot_performance()
    test.plot_return_dist()


if __name__ == '__main__':
    main() 