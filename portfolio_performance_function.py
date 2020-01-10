import numpy as np
import pandas as pd

def get_moving_std(data, window_size=60):
    result_array = list()
    
    if type(data) == pd.core.series.Series:
        data = np.asarray(data)
    
    for i in range(len(data) - window_size):
        result_array.append( data[i:i+window_size].std() )
        
    return pd.Series(result_array)

def get_moving_sharpe_ratio(data, window_size=60):
    result_array = list()
    
    if type(data) == pd.core.series.Series:
        data = np.asarray(data)
    
    for i in range(len(data) - window_size):
        result_array.append( data[i:i+window_size].mean() / data[i:i+window_size].std() )
        
    return pd.Series(result_array)

def get_moving_sortino_ratio(data, window_size = 60):
    result_array = []
    
    for i in range(len(data) - window_size):
        # Create a downside return column with the negative returns only
        data_in_range = data[i:i+window_size]
        
        downside_returns = (data_in_range[data_in_range < 0])

        # Calculate expected return and std dev of downside
        expected_return = data_in_range.mean()
        down_stdev = pd.Series(downside_returns).std()

        # Calculate the sortino ratio
        sortino_ratio = (expected_return - 0)/down_stdev
        result_array.append(sortino_ratio)
        
    return pd.Series(result_array)

def get_single_sortino_ratio(data):
    
    data = pd.Series(data)
    downside_returns = (data[data < 0])

    # Calculate expected return and std dev of downside
    expected_return = data.mean()
    down_stdev = pd.Series(downside_returns).std()

    # Calculate the sortino ratio
    sortino_ratio = (expected_return - 0)/down_stdev
    
    return sortino_ratio

def get_single_sharpe_ratio(data):
    
    # Calculate the sortino ratio
    sharpe_ratio = pd.Series(data).mean()/pd.Series(data).std()
    
    return sharpe_ratio

def get_single_downside_stdev(data):
    
    data = pd.Series(data)
    downside_returns = (data[data < 0])
    down_stdev = pd.Series(downside_returns).std()
    
    return down_stdev

def get_maximum_drawdown(daily_return_series):

    cum_ret = (daily_return_series+1).cumprod()
    running_max = np.maximum.accumulate(cum_ret)

    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1

    # Calculate the percentage drawdown
    drawdown = (cum_ret)/running_max - 1
    
    return drawdown.min()

def get_cvar_95(daily_return_series) : 
    
    var_95 = np.percentile(daily_return_series, 10)
    cvar_95 = daily_return_series[daily_return_series <= var_95].mean()
    return (cvar_95)

def plot_maximum_drawdown(daily_return_series):
    import matplotlib.pylab as plt
    
    import matplotlib.pylab as plt
    cum_ret = (daily_return_series+1).cumprod()
    running_max = np.maximum.accumulate(cum_ret)

    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1

    # Calculate the percentage drawdown
    drawdown = (cum_ret)/running_max - 1
    
    # Plot the results
    drawdown.plot()
    plt.show()
    
def get_cumulative_wealth(data):
    
    data = pd.Series(data)
    last_cumulative_wealth = list((data+1).cumprod())[-1]
    
    return last_cumulative_wealth