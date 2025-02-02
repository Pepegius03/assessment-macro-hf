import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from strategy import *
import time

class BacktestingEngine:
    """
    Backtesting engine for pairs trading strategies.

    :param strategy: The pairs trading strategy to backtest, an instance of PairsTradingStrategy.
    """
    def __init__(self, strategy: PairsTradingStrategy):
        self.strategy = strategy
        self.initial_aum = strategy.fund_aum
        
    
    def run_backtesting(self, end_date_backtest: pd.Timestamp = None, start_date_backtest: pd.Timestamp = None)-> pd.DataFrame:
        """
        Run the backtesting engine for the given strategy.

        :param end_date_backtest: The end date for the backtest.
        :param start_date_backtest: The start date for the backtest.
        :return: DataFrame containing the trade log.
        """
       
        self.strategy.trade_log = pd.DataFrame(columns=[ 'date', 'pos_1', 'pos_2', 'price_1', 'price_2', 'z_score', 'capital_trade', 'value_pos', 'aum', 'pnl_realized'])

        # we import the pickle file  with the trading days. Note that last trading day is 2025-01-31
        trading_days = pd.read_pickle('trading_days.pkl')
        filtered_trading_days = [day for day in trading_days if start_date_backtest <= day <= end_date_backtest]
        total_days = len(filtered_trading_days)

        for idx, day in enumerate(filtered_trading_days):
            if idx == 0:
                print(f"Starting backtest on {day}")
            self.strategy.end_date = day
            
            try:
                self.strategy.run(back_testing=True)
            except Exception as e:
                print(f"Error on {day}: {e}. Pausing for 100 seconds before continuing...")
                time.sleep(100)  
                continue  

            
            if (idx + 1) % 100 == 0 or idx == total_days - 1:
                print(f"Processed {idx + 1}/{total_days} trading days. Current date: {day}")

        return self.strategy.trade_log
    
    def compute_metrics(self, log_trade: pd.DataFrame) -> pd.DataFrame:
        """
        Compute performance metrics for the backtest.

        :param log_trade: DataFrame containing the trade log.
        :return: DataFrame containing the computed metrics.
        """
        total_pnl = log_trade['aum'].iloc[-1] - self.initial_aum
        
        total_return = (total_pnl / self.initial_aum) * 100  
        
        annualized_return = ((total_pnl / self.initial_aum) / (log_trade['date'].iloc[-1] - log_trade['date'].iloc[0]).days) * 365 * 100  # Convert to percentage
        
        log_trade['returns'] = np.where(
            (log_trade['capital_trade'].shift(1) != 0) & (log_trade['pnl_realized'] != 0),
            log_trade['pnl_realized'] / log_trade['capital_trade'].shift(1),
            0  
        )
        log_trade['returns'] = log_trade['returns'].fillna(0)
        
        daily_volatility = log_trade['returns'].std()
        annualized_volatility = daily_volatility * np.sqrt(252) * 100  

        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

        drawdown = log_trade['aum'] / log_trade['aum'].cummax() - 1
        max_drawdown = abs(drawdown.min()) * 100  

        metrics_df = pd.DataFrame({
            "Total PnL": [total_pnl],
            "Total Return (%)": [total_return],
            "Annualized Return (%)": [annualized_return],
            "Annualized Volatility (%)": [annualized_volatility],
            "Sharpe Ratio": [sharpe_ratio],
            "Max Drawdown (%)": [max_drawdown]
        })
    
        return metrics_df


    def plot_aum(self, log_trade: pd.DataFrame)-> plt.figure:
        """
        Plots the Assets Under Management (AUM) over time.

        :param log_trade: DataFrame containing 'date' and 'aum' columns.
        :return: The generated plot figure.
        """
        if log_trade is None or log_trade.empty:
            raise ValueError("Input DataFrame 'log_trade' is empty or None")

        required_cols = {'date', 'aum'}
        if not required_cols.issubset(log_trade.columns):
            raise ValueError(f"DataFrame must contain columns {required_cols}")

        log_trade['date'] = pd.to_datetime(log_trade['date']) 
        log_trade = log_trade.sort_values(by='date')  

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(log_trade['date'], log_trade['aum'], marker='o', linestyle='-', label='AUM')

        ax.set_xlabel('Date')
        ax.set_ylabel('AUM')
        ax.set_title('Assets Under Management Over Time')
        ax.legend()
        ax.grid(True)

        fig.autofmt_xdate()

        return fig


    def plot_cumulative_returns_with_positions(self, log_trade: pd.DataFrame, prices_columns: list, 
                                            pos_columns: list = None, start_date=None, end_date=None)-> plt.figure:
        """
        Generate a cumulative return plot with optional position markers.

        :param log_trade: DataFrame containing return columns and trade positions ('date', position columns).
        :param prices_columns: List of column names representing asset returns.
        :param pos_columns: List of position column names corresponding to assets.
        :param start_date: Optional start date for filtering data.
        :param end_date: Optional end date for filtering data.
        
        :return: Matplotlib figure
        """

        df = compute_simple_returns(log_trade, prices_columns, shift_days=1)

        df.rename(columns={'date': 'Date'}, inplace=True)
        df = filter_dataframe_by_date(df, start_date, end_date)

        df['cum_return_1'] = (1 + df['price_1_return']).cumprod()
        df['cum_return_2'] = (1 + df['price_2_return']).cumprod()

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df['Date'], df['cum_return_1'], label='Cumulative Return Price 1', color='blue')
        ax.plot(df['Date'], df['cum_return_2'], label='Cumulative Return Price 2', color='orange')

        ax.scatter(df['Date'][df[pos_columns[0]] > 0], df['cum_return_1'][df[pos_columns[0]] > 0],
                marker='^', color='green', label='Long Position ', s=100)
        ax.scatter(df['Date'][df[pos_columns[1]] > 0], df['cum_return_2'][df[pos_columns[1]] > 0],
                marker='^', color='green', s=100)

        ax.scatter(df['Date'][df[pos_columns[0]] < 0], df['cum_return_1'][df[pos_columns[0]] < 0],
                marker='v', color='red', label='Short Position', s=100)
        ax.scatter(df['Date'][df[pos_columns[1]] < 0], df['cum_return_2'][df[pos_columns[1]] < 0],
                marker='v', color='red', s=100)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Returns with Trading Positions')
        ax.legend()
        plt.xticks(rotation=45)
        plt.grid()

        return fig



