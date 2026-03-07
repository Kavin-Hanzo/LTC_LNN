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
        print(f"Fetching data for: {self.tickers}...")
        data = yf.download(
            self.tickers, 
            start=self.cfg['start_date'], 
            end=self.cfg['end_date'], 
            interval=self.cfg['interval']
        )['Close']
        return data.dropna()

    def plot_ticker_correlation(self, data):
        """Plots correlation matrix between different companies."""
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix: Inter-Company Closing Prices")
        
        path = os.path.join(self.save_dir, "company_correlation.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Inter-company correlation saved to {path}")

    def plot_feature_correlation(self, df, ticker_name):
        """Plots correlation between price and technical indicators for a single dataset."""
        # Add a few technical indicators for context
        df = df.copy()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Vol_Std'] = df['Close'].rolling(window=10).std()
        df = df.dropna()

        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='YlGnBu', fmt=".2f")
        plt.title(f"Feature Correlation Matrix: {ticker_name}")
        
        path = os.path.join(self.save_dir, f"feature_corr_{ticker_name}.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Feature correlation for {ticker_name} saved.")

    def plot_trends(self, data):
        """Plots normalized trend lines for all companies in a single graph."""
        # Normalize to 100 to compare growth regardless of price scale
        normalized_data = (data / data.iloc[0]) * 100
        
        plt.figure(figsize=(12, 6))
        for ticker in normalized_data.columns:
            plt.plot(normalized_data.index, normalized_data[ticker], label=ticker, linewidth=1.5)
            
        plt.title("Normalized Price Trends (Base 100)")
        plt.xlabel("Date")
        plt.ylabel("Relative Growth (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        path = os.path.join(self.save_dir, "price_trends.png")
        plt.savefig(path)
        plt.close()
        print(f"✅ Trend lines saved to {path}")

# Example Usage
if __name__ == "__main__":
    from Utils import load_config
    config = load_config("config.yaml")
    viz = DataVisualizer(config)
    
    # 1. Multi-company analysis
    multi_df = viz.fetch_multi_data()
    viz.plot_ticker_correlation(multi_df)
    viz.plot_trends(multi_df)
    
    # 2. Individual feature analysis (for the first ticker)
    # We fetch full OHLC for feature correlation
    single_ticker = viz.tickers[0]
    raw_df = yf.download(single_ticker, start=config['data']['start_date'], end=config['data']['end_date'])
    viz.plot_feature_correlation(raw_df, single_ticker)