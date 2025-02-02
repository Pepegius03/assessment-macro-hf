import numpy as np
import pandas as pd
import yfinance as yf
import datetime as date
from statsmodels.tsa.stattools import coint
from utils import *

class PairsTradingStrategy:
    def __init__(self, tickers: list, fund_aum: float = 50e6, stop_loss: float = -0.1, z_threshold: float = 2.2, 
                f_max: float = 0.8, z_max: float = 3.5, lookback_days: int = 60, end_date: pd.Timestamp = None):
        self.tickers = tickers
        self.fund_aum = fund_aum
        self.stop_loss = stop_loss
        self.z_threshold = z_threshold
        self.historical_data = None
        self.f_max = f_max  
        self.z_max = z_max
        self.lookback_days = lookback_days
        if not end_date:
            self.end_date = pd.Timestamp.today()
            print(f"End date not specified. Using today's date: {self.end_date}")
        self.end_date = pd.to_datetime(end_date)
        self.trade_log = pd.DataFrame(columns=[
            'date', 'pos_1', 'pos_2', 
            'price_1', 'price_2', 'z_score', 'capital_trade', 'value_pos', 'aum', 'pnl_realized'])
    
    def load_data(self):
        """ 
        Load historical data for the selected tickers assigning to the end date the last date of the historical_data, keeping the end_date exclusive
        """
        self.create_historical_data()
        self.end_date = self.historical_data['Date'].iloc[-1]

    def create_historical_data(self):
        """ 
        Create historical data for the selected tickers from end_date - lookback_days to end_date
        """
        end_date = self.end_date
        start_date = end_date - pd.Timedelta(days=self.lookback_days)
        historical_data = download_data_yfinance(self.tickers, start_date, end_date)
        selected_columns = ["Date"] + [f"Close_{ticker}" for ticker in self.tickers]
        self.historical_data = process_yfinance_data(historical_data, self.tickers, selected_columns=selected_columns)

    
    def calculate_z_score(self, ols_results: dict) -> float:
        """
        Calculate the z-score for the residuals of the OLS regression
        
        :param ols_results: Dictionary with OLS regression results
        :return: Z-score
        """
        residuals = ols_results['residuals']
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)
        z_score = (residuals.iloc[-1] - mean_resid) / std_resid
        return z_score
    
    def get_kelly_allocation(self, z_score: float) -> float:
        """
        Dynamically allocate capital using a scaled Kelly Criterion approach
        
        :param z_score: Z-score of the residuals
        :return: Capital per trade
        """
        if abs(z_score) < self.z_threshold:
            return 0  

        f_star = min(abs(z_score) / self.z_max, 1) * self.f_max  
        capital_per_trade = f_star * self.fund_aum
        
        return capital_per_trade


    def execute_trade(self, z_score: float, dates: pd.Timestamp = None, columns: list = None):
        """
        Execute a trade based on the z-score of the residuals updating the trade log.

        :param z_score: Z-score of the residuals
        :param columns: List of column names for the prices of the two assets

        """
        
        if not dates:
            dates = self.end_date

        if dates not in self.historical_data['Date'].values:
            raise ValueError(f"Date {dates} not found in historical data")
        
        if not columns:
            raise ValueError("Columns must be specified")
        

        capital_per_trade = self.get_kelly_allocation(z_score)
        
        if z_score < 0:
            pos_long = capital_per_trade / self.historical_data[columns[0]].iloc[-1]
            pos_short = -capital_per_trade / self.historical_data[columns[1]].iloc[-1]
            new_trade = pd.DataFrame([{
                        'date': dates,
                        'pos_1': pos_long, 
                        'pos_2': pos_short,
                        'price_1': self.historical_data.loc[self.historical_data['Date'] == dates, columns[0]].values[0],
                        'price_2': self.historical_data.loc[self.historical_data['Date'] == dates, columns[1]].values[0],
                        'z_score': z_score,
                        'capital_trade': capital_per_trade
                    }])
            self.trade_log = pd.concat([self.trade_log, new_trade], ignore_index=True)
        else:
            pos_long = capital_per_trade / self.historical_data[columns[1]].iloc[-1]
            pos_short = -capital_per_trade / self.historical_data[columns[0]].iloc[-1]
            new_trade = pd.DataFrame([{
                        'date': dates,
                        'pos_1': pos_short, 
                        'pos_2': pos_long,
                        'price_1': self.historical_data.loc[self.historical_data['Date'] == dates, columns[0]].values[0],
                        'price_2': self.historical_data.loc[self.historical_data['Date'] == dates, columns[1]].values[0],
                        'z_score': z_score,
                        'capital_trade': capital_per_trade
                    }])
            self.trade_log = pd.concat([self.trade_log, new_trade], ignore_index=True)


    def compute_value(self):
        """
        Updates the trade log with the realized PnL, the value of the position and the AUM.
        """
        if len(self.trade_log) == 0:
            raise ValueError("No trades executed yet")
            
        elif len(self.trade_log) == 1:
            print("Only one trade executed.")
            self.trade_log.loc[self.trade_log.index[-1], 'pnl_realized'] = 0.0
            self.trade_log.loc[self.trade_log.index[-1], 'aum'] = self.fund_aum
            self.trade_log.loc[self.trade_log.index[-1], 'value_pos'] = self.trade_log['pos_1'].iloc[0] * self.trade_log['price_1'].iloc[0] + self.trade_log['pos_2'].iloc[0] * self.trade_log['price_2'].iloc[0]
        
        else:
            index_t = len(self.trade_log) - 1
            index_t_minus_1 = len(self.trade_log) - 2
            pnl_realized = self.trade_log['pos_1'].iloc[index_t_minus_1] * self.trade_log['price_1'].iloc[index_t] - self.trade_log['pos_1'].iloc[index_t] * self.trade_log['price_1'].iloc[index_t] + self.trade_log['pos_2'].iloc[index_t_minus_1] * self.trade_log['price_2'].iloc[index_t] - self.trade_log['pos_2'].iloc[index_t] * self.trade_log['price_2'].iloc[index_t]
            self.trade_log.loc[self.trade_log.index[-1], 'pnl_realized']  = pnl_realized
            
            value_today = self.trade_log['pos_1'].iloc[index_t] * self.trade_log['price_1'].iloc[index_t] + self.trade_log['pos_2'].iloc[index_t] * self.trade_log['price_2'].iloc[index_t]
            self.trade_log.loc[self.trade_log.index[-1], 'value_pos'] = value_today
            
            self.fund_aum += pnl_realized
            self.trade_log.loc[self.trade_log.index[-1], 'aum'] = self.fund_aum
        

    def run(self, back_testing: bool = False):
        """
        Run a complete pairs trading strategy.

        :param back_testing: If True, loads historical data from a CSV file. If False, loads live data using self.load_data(). Use True for backtesting (way faster) and False for live trading.
        """
        if back_testing:
        # Load data from CSV file for backtesting
            self.historical_data = pd.read_csv('historical_data.csv')
            self.historical_data = filter_dataframe_by_date(
                self.historical_data, 
                start_date=self.end_date - pd.Timedelta(days=self.lookback_days), 
                end_date=self.end_date)
            end_date = self.end_date 

        else:
            self.load_data()

        start_date = self.historical_data['Date'].iloc[0]
        
        if self.stop_loss_bool():
            z_score = 0
            self.execute_trade(z_score, columns=['Close_GC=F', 'Close_SI=F'])
            self.compute_value()
        else:
        
            ols_results = ols_regression(self.historical_data, "Close_GC=F", "Close_SI=F",
                                        start_date=start_date.strftime("%Y-%m-%d"),
                                        end_date=self.end_date.strftime("%Y-%m-%d"))
            z_score = self.calculate_z_score(ols_results)
            self.execute_trade(z_score, columns=['Close_GC=F', 'Close_SI=F'])
            
            self.compute_value()
        

    def stop_loss_bool(self)-> bool:
        """
        Check if the stop loss condition is met.

        :return: True if the stop loss condition is met, False otherwise.
        """

        if len(self.trade_log) == 0 or len(self.trade_log) == 1:
            return False
    
        prev_capital_trade = self.trade_log['capital_trade'].shift(1).iloc[-1]
        pnl_realized = self.trade_log['pnl_realized'].iloc[-1]

        if np.isnan(prev_capital_trade) or np.isnan(pnl_realized):
            last_return = 0
        elif (prev_capital_trade != 0) and (pnl_realized != 0):
            last_return = pnl_realized / prev_capital_trade
        else:
            last_return = 0

        return last_return < self.stop_loss

