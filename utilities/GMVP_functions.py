from . import prepare_data
from . import bounded_Kmeans_clustering as bounded

import numpy as np
import pandas as pd

# libraries for Bounded K-means clustering
import geojsonio
import json

import time
from geojson import Point as GeojsonPoint, Feature, FeatureCollection
from shapely.geometry import Point as ShapelyPoint, Polygon
from sklearn.datasets import make_blobs
    

# libraries related with normalization
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

# libraries related with plotting
import seaborn as sns
import matplotlib.pyplot as plt

# libraries for using PCA
from sklearn.decomposition import PCA

# libraries for using t-sne
from sklearn.manifold import TSNE

#--------------------------------------------------------------------------------------------------------------------------------------#

def GMVP_within_cluster(data_period, index, stock_list_of_cluster):
    
    file_path = './data/russel1000_daily_price_df.pickle'
    daily_return_df_list_val, daily_return_df_list_test, daily_price_df, all_ticker_list = prepare_data.split_dataset(file_path)
    
    if data_period == 'validation':
        daily_return_df_list = daily_return_df_list_val
    elif data_period == 'test':
        daily_return_df_list = daily_return_df_list_test
    
    # slice the data needed
    daily_stock_return_of_cluster_df = daily_return_df_list[index].loc[:, stock_list_of_cluster]
    
    # get the covariance matrix and inverse matrix of covariance matrix respectively
    cov_mat_of_stock_within_cluster_df = daily_stock_return_of_cluster_df.cov()
    cov_mat_of_stock_within_cluster_array = cov_mat_of_stock_within_cluster_df.values
    inv_cov_mat_array = np.linalg.pinv(cov_mat_of_stock_within_cluster_array) # Use pseudo-inverse incase matrix is singular / ill-conditioned

    # construct minimum variance weights
    one_vector_array = np.ones(len(inv_cov_mat_array))
    inv_dot_one_array = np.dot(inv_cov_mat_array, one_vector_array)
    within_cluster_weight_array = inv_dot_one_array/ np.dot(inv_dot_one_array, one_vector_array)
    
    # calculate the daily_return
    daily_weighted_stock_return_df = (daily_stock_return_of_cluster_df * within_cluster_weight_array)
    daily_cluster_return_df = daily_weighted_stock_return_df.sum(axis=1)
    
    return daily_cluster_return_df, within_cluster_weight_array

#--------------------------------------------------------------------------------------------------------------------------------------#

def GMVP_between_clusters(data_period, max_cluster_size, scaling_method='none', dim_reduction_method='none', no_of_PCA_components = 3, no_of_tsne_components = 3):
    rebalancing_period = 60
    
    ################## creating lists and dictionaries for storing outcomes ##################
    # 1) returns
    daily_portfolio_return_list = []   # daily returns of portfolio
    # 2) stdev
    in_sample_stdev_list = []          # standard deviation of in-sample (252-day-long) daily portfolio returns
    out_of_sample_stdev_list = []      # standard deviation of out-of-sample (60-day-long) daily portfolio returns
    # 3) cluster & stock 
    cluster_and_stock_dict = {}        # 1. cluster_return : 252_daily_returns * 11_clusters
                                       # 2. cluster_weight : weights of each 11 cluster
                                       # 3. cluster_ticker : tickers belonging to each cluster 
                                       # 4. stock_weight   : weights of each individual stock
                                       # 5. daily_return_for_viz 
    ##########################################################################################
    
    
    file_path = './data/russel1000_daily_price_df.pickle'
    daily_return_df_list_val, daily_return_df_list_test, daily_price_df, all_ticker_list = prepare_data.split_dataset(file_path)
    
    if data_period == 'validation':
        daily_return_df_list = daily_return_df_list_val
    elif data_period == 'test':
        daily_return_df_list = daily_return_df_list_test
    for index_no, daily_return_df in enumerate(daily_return_df_list):
        print(f"code working: {index_no}/{len(daily_return_df_list)} done")
        
        cluster_weight_dict = {}   # weights of each 11 cluster
        cluster_ticker_dict = {}   # tickers belonging to each cluster 
        stock_weight_within_cluster_dict = {} # weights of each individual stock
        
        ################## 1. normalizing data ##################
        if scaling_method == 'standard_scale':    
            scaled_daily_price_array = (standard_scaler.fit_transform(daily_return_df))
            after_scaling_return_df = pd.DataFrame(scaled_daily_price_array, columns = all_ticker_list).T
        elif scaling_method == 'none':
            after_scaling_return_df = daily_return_df.T
        
        ################## 2. dimensionality reduction ##################
        if dim_reduction_method == 'PCA':
            pca = PCA(n_components=no_of_PCA_components)
            scaled_daily_return_PCA_array = pca.fit_transform(after_scaling_return_df)
            after_dim_reduction_return_df = pd.DataFrame(scaled_daily_return_PCA_array, index = all_ticker_list)  # shape : [stocks * PCs]
        elif dim_reduction_method == 'tsne':
            tsne = TSNE(n_components = no_of_tsne_components)
            scaled_daily_return_tsne_cudf = tsne.fit_transform(after_scaling_return_df.values)
            after_dim_reduction_return_df = pd.DataFrame(scaled_daily_return_tsne_cudf, index = all_ticker_list)
        elif dim_reduction_method == 'none':
            after_dim_reduction_return_df = after_scaling_return_df
            
        ################## 3. bounded k-means clustering  ################## 
        n_clusters = 11
        n_iter = 30
        n_init = 15

        weights = np.ones(after_dim_reduction_return_df.shape[0]) # The original code is created to consider the observations' weights, which is not needed in our research.
                                                                  # As we judge stocks solely based on return movements while coming up with a portfolio, we assign 1 to every stock.

        cluster_maker = bounded.BoundedKMeansClustering(n_clusters, max_cluster_size, n_iter, n_init)
        best_cost, best_clusters = cluster_maker.fit(after_dim_reduction_return_df.values, weights)
        after_dim_reduction_return_df.loc[:,'cluster_label'] = 0  # assign false classification 0 at first, but correctly classifies stocks right after. 
        
        for cluster_label in range(n_clusters):
            ticker_index_list =  best_clusters[cluster_label]
            cluster_ticker_list = list(np.array(all_ticker_list)[ticker_index_list])
            after_dim_reduction_return_df.loc[cluster_ticker_list,'cluster_label'] = cluster_label
            
        # storing clustering results in a dict : cluster_ticker_dict
        cluster_label_list = list((after_dim_reduction_return_df.loc[:,'cluster_label'].unique()))
        for cluster_label in cluster_label_list:
            cluster_ticker_dict[cluster_label] = list(after_dim_reduction_return_df[after_dim_reduction_return_df.loc[:,'cluster_label'] == cluster_label].index)
        
        ################## (Optional) For visualization ##################
        pca = PCA(n_components=2)
        daily_return_array_for_viz = pca.fit_transform(after_dim_reduction_return_df.iloc[:,:-1])
        daily_return_df_for_viz = pd.DataFrame(daily_return_array_for_viz, index = all_ticker_list)  # shape : [stocks * PCs]
        daily_return_df_for_viz.loc[: ,'cluster_label'] = after_dim_reduction_return_df.loc[:,'cluster_label']
        daily_return_df_for_viz.rename(columns = {0: 'PC_1',1:'PC_2'}, inplace=True)
        
        # --------------------- #
        #  GMVP within cluster  #
        # --------------------- #
        # GMVP on each cluster using the function 'GMVP_within_cluster' --> to create a matrix of daily returns of clusters
        stock_weight_within_cluster_dict = {}
        daily_cluster_return_dict = {}
        
        ################## 4. computing stock weights within a cluster ##################
        for cluster_label in cluster_label_list:
            daily_cluster_return_series, stock_weight_within_cluster_array = GMVP_within_cluster(data_period, index_no, cluster_ticker_dict[cluster_label])
            daily_cluster_return_dict[cluster_label] = daily_cluster_return_series
            stock_weight_within_cluster_dict[cluster_label] = stock_weight_within_cluster_array

        daily_cluster_return_df = pd.DataFrame.from_dict(daily_cluster_return_dict)
        
        # ---------------------- #
        #  GMVP between clusters #
        # ---------------------- #
        cov_mat_of_cluster_df = daily_cluster_return_df.cov()
        cov_mat_of_cluster_array = cov_mat_of_cluster_df.values
        inv_cov_mat_array = np.linalg.pinv(cov_mat_of_cluster_array) # Use pseudo-inverse incase matrix is singular / ill-conditioned

        ################## 5. computing cluster weights ##################
        one_vector_array = np.ones(len(inv_cov_mat_array))
        inv_dot_one_array = np.dot(inv_cov_mat_array, one_vector_array)
        cluster_weight_array = inv_dot_one_array/ np.dot( inv_dot_one_array , one_vector_array)
        cluster_weight_df = pd.DataFrame(data= cluster_weight_array, columns = ['weight'], index = cluster_label_list)
        
        # compute stdev of portfolio, which can be calculated from daily returns of clusters
        in_sample_variance = np.dot(cluster_weight_array, np.dot(cov_mat_of_cluster_array, cluster_weight_array))
        in_sample_stdev = np.sqrt(in_sample_variance)
        
        ################## 6. computing stock weights in a portfolio (portfolio weights) ##################
        temp_portfolio_weight_list = []

        for cluster_label in cluster_label_list:
            stock_weight_within_sector_array =  cluster_weight_df.loc[cluster_label,'weight'] * stock_weight_within_cluster_dict[cluster_label]
            stock_weight_within_sector_df = pd.DataFrame(stock_weight_within_sector_array, index = cluster_ticker_dict[cluster_label], columns = ['weight'])
            temp_portfolio_weight_list.append(stock_weight_within_sector_df)

        portfolio_weight_df = pd.concat(temp_portfolio_weight_list)
        
        # ---------------------------------------------------------- #
        # Calculating daily return based on GMV optimization results #
        # ---------------------------------------------------------- #
        # we should use 'index+1' ; we optimize portfolio at the time point 'index' and see how it goes for the time period from 'index' to 'index+1'
        if (index_no+1) < len(daily_return_df_list):
            
            # appending to the list only if we invest in the market, following the portfolio optimization
            in_sample_stdev_list.append(in_sample_stdev)
            
            # appending data only if the data is used for calculating next time's return 
            future_daily_return_df = daily_return_df_list[index_no+1]
            future_daily_cluster_return_dict = {}

            for cluster_label in cluster_label_list:

                # 1st optimization - cluster_weight calculated from 'between_cluster GMV'
                cluster_weight = cluster_weight_df.loc[cluster_label,'weight']
                # 2nd optimization - stock_weight calculated from 'inside_cluster GMV'
                stock_weight_within_cluster_array = stock_weight_within_cluster_dict[cluster_label]
                # based on the asset allocation reuslt from 1st optimization and 2nd optimization, now we can compute the each stock's weight
                future_daily_cluster_return_df = (future_daily_return_df.loc[:, cluster_ticker_dict[cluster_label]] * stock_weight_within_cluster_array * cluster_weight)
                future_daily_cluster_return_series = future_daily_cluster_return_df.sum(axis=1)[-rebalancing_period:]
                future_daily_cluster_return_dict[cluster_label] = future_daily_cluster_return_series

            # out-of-sample daily portfolio returns (60 days)
            future_daily_cluster_return_df = pd.DataFrame.from_dict(future_daily_cluster_return_dict)
            daily_portfolio_return_series_60days = future_daily_cluster_return_df.sum(axis=1)
            daily_portfolio_return_list.extend(daily_portfolio_return_series_60days)
            
            # standard deviation of out-of-sample portfolio returns
            out_of_sample_stdev = daily_portfolio_return_series_60days.std()
            out_of_sample_stdev_list.append(out_of_sample_stdev)
            
            # saving outcomes to a dictionary
            cluster_and_stock_dict[index_no] = {'cluster_return' : daily_cluster_return_df,
                                                'cluster_weight' : cluster_weight_df,
                                                'cluster_ticker' : cluster_ticker_dict,
                                                'stock_weight' : portfolio_weight_df,
                                                'daily_return_for_viz' : daily_return_df_for_viz
                                                }
            
    daily_portfolio_return_array = np.array(daily_portfolio_return_list)
    in_sample_stdev_series = pd.Series(in_sample_stdev_list)
    out_of_sample_stdev_series = pd.Series(out_of_sample_stdev_list)
    
    return daily_portfolio_return_array, in_sample_stdev_series, out_of_sample_stdev_series, cluster_and_stock_dict