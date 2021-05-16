import pandas as pd

def split_dataset(file_path):
    
    daily_price_df = pd.read_pickle(file_path)
    daily_price_df.index.name = None
    
    all_ticker_list = list(daily_price_df.columns)
    
    daily_return_df = daily_price_df.pct_change()
    daily_return_df = daily_return_df.dropna(axis=0)
    
    # past 252 days' daily returns are considered to calculate the covariance
    seq_length = 252

    # portfolio rebalnacing period is 60 days
    rebalancing_period = 60

    # sliced dataframes are stored in the list below temporarily
    sliced_daily_return_df_list = []

    for i in range(0, (daily_return_df.shape[0]-(seq_length)+1), rebalancing_period):  # i gets bigger by 20 : 0, 20, 40,...
        sliced_daily_return_df = (daily_return_df.iloc[(i):(i+seq_length),:])
        sliced_daily_return_df_list.append(sliced_daily_return_df) # this is used for portfolio optimization, so can stay as pd.DataFrame
    
    list_length = len(sliced_daily_return_df_list)
    
    # splitting 'sliced dataframes' into two different lists : validation and test
    daily_return_df_list_val = []
    daily_return_df_list_test = []

    # for validation
    for i in range(len(sliced_daily_return_df_list)):
        if i < (int(list_length * 0.6)):
            daily_return_df_list_val.append(sliced_daily_return_df_list[i])
        elif i >= ((int(list_length * 0.6))+4) : 
            daily_return_df_list_test.append(sliced_daily_return_df_list[i])
            
    return daily_return_df_list_val, daily_return_df_list_test, daily_price_df, all_ticker_list