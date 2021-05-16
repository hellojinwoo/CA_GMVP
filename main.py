from utilities import prepare_data 
from utilities import GMVP_functions
from utilities import portfolio_performance_functions as perf

import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_period', type=str)  # data_period = 'validation' or 'test'
parser.add_argument('--max_cluster_size', type=int) # max_cluster_size = maximum number of stocks that can be put in a single cluster 
parser.add_argument('--scaling_method', type=str) # scaling_method = 'standard_scaled' or 'none'
parser.add_argument('--dim_reduction_method', type=str) # dim_reduction_method = 'PCA', 'tsne', or 'none'
parser.add_argument('--PCA_components', type=int, default=3)
parser.add_argument('--tsne_components', type=int, default=3)

args = parser.parse_args()

if __name__ == "__main__":
    #########################
    # 1. loading stock data #
    #########################
    file_path = './data/russel1000_daily_price_df.pickle'
    daily_return_df_list_val, daily_return_df_list_test, daily_price_df, all_ticker_list = prepare_data.split_dataset(file_path)
    
    validation_first_date =  str(daily_return_df_list_val[1].index[-60].date())
    validation_last_date = str(daily_return_df_list_val[-1].index[-1].date())
    validation_investment_date = daily_price_df.loc[validation_first_date:validation_last_date,:].index

    test_first_date =  str(daily_return_df_list_test[1].index[-60].date())
    test_last_date = str(daily_return_df_list_test[-1].index[-1].date())
    test_investment_date_validation = daily_price_df.loc[test_first_date:test_last_date,:].index
    
    ##########################################
    # 2. performing a portfolio optimization #
    ##########################################
    
    portfolio_return, in_sample_std, out_of_sample_std, cluster_and_stock_dict = GMVP_functions.GMVP_between_clusters(data_period = args.data_period,
                                                                                                                      max_cluster_size = args.max_cluster_size,
                                                                                                                      scaling_method = args.scaling_method,
                                                                                                                      dim_reduction_method = args.dim_reduction_method,
                                                                                                                      no_of_PCA_components = args.pca_components,
                                                                                                                      no_of_tsne_components = args.tsne_components)
    # in-sample_std
    print(f"annualized in-sample stdev : {in_sample_std.mean()*np.sqrt(252):.4f}")
    # out-of-sample_std
    print(f"annualized out-of-sample stdev : {out_of_sample_std.mean()*np.sqrt(252):.4f}")
    print('------------------------------')
    print(f"annualized_sharpe_ratio : {perf.get_single_sharpe_ratio(portfolio_return)*np.sqrt(252):.4f}")
    print(f"annualized_sortino_ratio : {perf.get_single_sortino_ratio(portfolio_return)*np.sqrt(252):.4f}")
    print('------------------------------')
    # downside_std
    print(f"annaulized downside_stdev : {perf.get_single_downside_stdev(portfolio_return)*np.sqrt(252):.4f}")
    # MDD
    print(f"MDD : {perf.get_maximum_drawdown(portfolio_return)*100:.2f}%")
    # CVaR_95
    print(f"CVaR_95 : {perf.get_cvar_95(portfolio_return)*100:.2f}%")
