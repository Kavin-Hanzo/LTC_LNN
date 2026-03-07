import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class DataVisualizer:
    def __init__(self, config, save_dir="./experiments/plots"):
        self.cfg = config['data']
        self.tickers = self.cfg.get('tickers', [self.cfg['ticker']])
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def fetch_multi_data(self):
        """Downloads Close prices for all tickers in config."""
        print(f"📥 Fetching data for: {self.tickers}...")
        data = yf.download(
            self.tickers, 
            start=self.cfg['start_date'], 
            end=self.cfg['end_date'], 
            interval=self.cfg['interval'],
            progress=False
        )['Close']
        
        # Ensure it's a DataFrame even if only one ticker is fetched
        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.tickers[0])
            
        return data.dropna()

    def plot_price_distributions(self, data):
        """
        Plots the Histogram + KDE (Kernel Density Estimate) for each stock.
        Reveals skewness and volatility 'fat tails'.
        """
        num_stocks = len(data.columns)
        fig, axes = plt.subplots(num_stocks, 1, figsize=(10, 4 * num_stocks))
        
        # Flatten axes if there's more than one, otherwise put in list
        if num_stocks == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, ticker in enumerate(data.columns):
            sns.histplot(data[ticker], kde=True, ax=axes[i], color='teal', bins=40)
            axes[i].set_title(f"Price Value Distribution: {ticker}", fontsize=14, fontweight='bold')
            axes[i].set_xlabel("Closing Price (USD)")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, "price_distributions.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✅ Price distributions saved to {path}")

    def plot_ticker_correlation(self, data):
        """Correlation matrix between different companies."""
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Inter-Company Price Correlation Matrix", fontweight='bold')
        
        path = os.path.join(self.save_dir, "company_correlation.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Inter-company correlation saved to {path}")

    def plot_trends(self, data):
        """Normalized trend lines (Base 100) for performance comparison."""
        # Normalize: (Price / First Price) * 100
        normalized_data = (data / data.iloc[0]) * 100
        
        plt.figure(figsize=(12, 6))
        for ticker in normalized_data.columns:
            plt.plot(normalized_data.index, normalized_data[ticker], label=ticker, linewidth=2)
            
        plt.title("Normalized Growth Trends (Base 100 Index)", fontsize=14, fontweight='bold')
        plt.xlabel("Timeline")
        plt.ylabel("Relative Growth (%)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        path = os.path.join(self.save_dir, "normalized_trends.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Trend lines saved to {path}")

# Integration helper
def run_full_eda(config):
    viz = DataVisualizer(config)
    data = viz.fetch_multi_data()
    
    viz.plot_price_distributions(data)
    viz.plot_ticker_correlation(data)
    viz.plot_trends(data)