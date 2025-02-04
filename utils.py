import yfinance as yf
import datetime as date
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def download_data_yfinance(tickers: list, start_date: date.date, end_date: date.date) -> dict:
    """
    Download historical data from Yahoo Finance

    :param tickers: list of tickers
    :param start_date: start date
    :param end_date: end date
    :return: dictionary with historical data for each ticker
    """
    historical_data = {}

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        historical_data[ticker] = data

    return historical_data

def process_yfinance_data(historical_data: dict, tickers: list, merge_on: str = "Date", selected_columns: list = None) -> pd.DataFrame:
    """
    Process historical data from Yahoo Finance for multiple tickers and merge them.

    :param historical_data: Dictionary with historical data for each ticker.
    :param tickers: List of tickers to process.
    :param merge_on: Column name to merge dataframes on.
    :param selected_columns: List of columns to keep after merging. Defaults to all.
    :return: Merged DataFrame with selected columns.
    """
    if not tickers or len(tickers) < 2:
        raise ValueError("At least two tickers are required for merging")
    
    dataframes = {}
    for ticker in tickers:
        df = historical_data.get(ticker, pd.DataFrame()).copy()
        if df.empty:
            raise ValueError(f"Data for ticker {ticker} is missing or empty")
        df.columns = df.columns.get_level_values(0)
        df.columns.name = None
        df.reset_index(inplace=True)
        dataframes[ticker] = df
    
    merged_df = dataframes[tickers[0]]
    for ticker in tickers[1:]:
        merged_df = pd.merge(merged_df, dataframes[ticker], on=merge_on, suffixes=(f"_{tickers[0]}", f"_{ticker}"))
    
    if selected_columns:
        merged_df = merged_df[selected_columns]
    
    merged_df[merge_on] = pd.to_datetime(merged_df[merge_on])
    
    return merged_df

def compute_simple_returns(df: pd.DataFrame, price_columns: list, shift_days: int = 1) -> pd.DataFrame:
    """
    Compute simple returns for specified price columns.

    :param df: Input DataFrame containing price columns.
    :param price_columns: List of column names representing asset prices.
    :param shift_days: Number of days to shift for return calculation.
    :return: DataFrame containing simple returns.
    """
    if not price_columns:
        raise ValueError("At least one price column must be provided")
    
    returns_df = df.copy()
    
    for col in price_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
        
        returns_df[f"{col}_return"] = df[col].pct_change(periods=shift_days)
    
    return returns_df[["Date"] + [f"{col}_return" for col in price_columns]] if "Date" in df.columns else returns_df

def filter_dataframe_by_date(df: pd.DataFrame, start_date, end_date, exclusive: bool = False) -> pd.DataFrame:
    """
    Filter a DataFrame based on a date range.

    :param df: Input DataFrame containing a 'Date' column.
    :param start_date: Start date for filtering (inclusive).
    :param end_date: End date for filtering (inclusive or exclusive).
    :param exclusive: If True, exclude the end date.
    :return: Filtered DataFrame.
    """
    df['Date'] = pd.to_datetime(df['Date'])

    if exclusive:
        return df[(df['Date'] >= start_date) & (df['Date'] < end_date)].copy()
    
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()



def plot_cumulative_returns(df: pd.DataFrame, return_columns: list, start_date=None, end_date=None)  -> plt.figure:
    """
    Generate a cumulative return plot for specified return columns and return the plot object.

    :param df: Input DataFrame containing return columns.
    :param return_columns: List of column names representing asset returns.
    :param start_date: Optional start date for filtering data.
    :param end_date: Optional end date for filtering data.
    
    :return: Matplotlib figure
    """
    if start_date and end_date:
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    cum_return_df = df.copy()
    for col in return_columns:
        cum_return_df[f"{col}_cum"] = (1 + cum_return_df[col]).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in return_columns:
        ax.plot(cum_return_df["Date"], cum_return_df[f"{col}_cum"], label=f"{col} Cumulative Return", linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Returns Over Time")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

    return fig  

def ols_regression(df: pd.DataFrame, x_col: str, y_col: str, start_date: date.date, end_date: date.date) -> dict:
    """
    Perform an Ordinary Least Squares (OLS) regression.

    :param df: Input DataFrame containing price columns.
    :param x_col: Name of the first price column.
    :param y_col: Name of the second price column.
    :param start_date: Optional start date for filtering data.
    :param end_date: Optional end date for filtering data.

    :return: Dictionary with alpha, beta, and residuals.
    """
    df_with_today = df.copy()
    if start_date and end_date:
        df_with_today = filter_dataframe_by_date(df, start_date, end_date, exclusive=False)

    if start_date and end_date:
        df = filter_dataframe_by_date(df, start_date, end_date, exclusive=True)
    
    X = sm.add_constant(df[x_col])
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    
    theoretical_y = model.params['const'] + model.params[x_col] * df_with_today[x_col]
    today_residual = df_with_today[y_col].iloc[-1] - theoretical_y.iloc[-1]
    
    return {
        'alpha': model.params['const'],
        'beta': model.params[x_col],
        'residuals': model.resid,
        'today_residual': today_residual
    }


def engle_granger_cointegration_test(df: pd.DataFrame, x_col: str, y_col: str, start_date: date.date, end_date: date.date, p_threshold: float = 0.2) -> bool:
    """
    Perform the Engle-Granger cointegration test for two price series.

    :param df: Input DataFrame containing price columns.
    :param x_col: Name of the first price column.
    :param y_col: Name of the second price column.
    :return: True if the series are cointegrated, False otherwise.
    """

    models = ols_regression(df, x_col, y_col, start_date, end_date)
    test_result = adfuller(models['residuals'])
    
    return test_result[1] < p_threshold, models



