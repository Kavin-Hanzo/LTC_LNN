import os
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    def __init__(self, config):
        """
        Initializes with config.yaml structure:
        data_config -> historical_sets, tickers
        logging -> save_dir
        """
        self.data_cfg = config['data_config']
        self.tickers = self.data_cfg['tickers']
        self.sets = self.data_cfg['historical_sets']
        
        # Setup directories
        self.base_dir = config['logging']['save_dir']
        self.plot_dir = os.path.join(self.base_dir, "eda_plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def fetch_set_data(self, set_info):
        """Downloads all tickers for a specific historical set."""
        print(f"📥 Fetching data for {set_info['name']} ({set_info['interval']})...")
        data = yf.download(
            self.tickers,
            start=set_info['start'],
            end=set_info['end'],
            interval=set_info['interval'],
            progress=False
        )['Close']
        
        # Handle cases where yfinance returns a Series for single ticker
        if len(self.tickers) == 1:
            data = data.to_frame(name=self.tickers[0])
            
        return data.dropna()

    def plot_correlations(self, data, set_name):
        """Plots the correlation matrix for the specific set of tickers."""
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Matrix: {set_name}", fontsize=14, fontweight='bold')
        
        save_path = os.path.join(self.plot_dir, f"{set_name}_correlation.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Correlation Plot: {save_path}")

    def plot_normalized_trends(self, data, set_name):
        """Plots trend lines for all tickers on a single graph (Normalized to Base 100)."""
        # Normalize: (Price / Initial Price) * 100
        normalized = (data / data.iloc[0]) * 100
        
        plt.figure(figsize=(12, 6))
        for ticker in normalized.columns:
            plt.plot(normalized.index, normalized[ticker], label=ticker, linewidth=1.5)
            
        plt.title(f"Normalized Price Trends: {set_name} (Base 100)", fontsize=14, fontweight='bold')
        plt.xlabel("Timeline")
        plt.ylabel("Growth (%)")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plot_dir, f"{set_name}_trends.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Trend Plot: {save_path}")

    def plot_value_distributions(self, data, set_name):
        """Plots Histogram + KDE for each ticker in the set."""
        num_tickers = len(data.columns)
        fig, axes = plt.subplots(num_tickers, 1, figsize=(10, 4 * num_tickers))
        
        if num_tickers == 1:
            axes = [axes]

        for i, ticker in enumerate(data.columns):
            sns.histplot(data[ticker], kde=True, ax=axes[i], color='teal', bins=40)
            axes[i].set_title(f"{ticker} Distribution ({set_name})", fontsize=12)
            axes[i].set_xlabel("Price (USD)")
            axes[i].grid(axis='y', alpha=0.2)

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, f"{set_name}_distributions.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Distribution Plot: {save_path}")

    def run_all_eda(self):
        """Iterates through all sets in config and generates plots."""
        for set_info in self.sets:
            set_name = set_info['name']
            df = self.fetch_set_data(set_info)
            
            if df.empty:
                print(f"⚠️ No data found for {set_name}, skipping...")
                continue
                
            self.plot_correlations(df, set_name)
            self.plot_normalized_trends(df, set_name)
            self.plot_value_distributions(df, set_name)

# --- Usage Example ---
if __name__ == "__main__":
    import yaml
    
    # Load your config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    visualizer = DataVisualizer(config)
    visualizer.run_all_eda()