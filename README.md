# Clustering Approaches for GMVP

- This is the source code used for experiments for the research paper "<a href = "https://arxiv.org/abs/2001.02966">__Clustering Approaches for Global Minimum Variance Portfolio__"</a>
- The academic paper utilizes 'constrained K-means clustering' to group stocks showing similar price movements before performing 'within group portfolio optimization'. 

## Example
- Using raw data without scaling methods and dimensional reduction methods
```
python main.py --data_period test --max_cluster_size 75 --scaling_method none --dim_reduction_method none
```
- Using PCA without scaling methods (If no no_of_PCA_components is specfied, the default number 3 is used) 
```
python main.py --data_period test --max_cluster_size 75 --scaling_method none --dim_reduction_method PCA
```
- Using t-sne with standard scaling and t-sne components = 10
```
python main.py --data_period test --max_cluster_size 75 --scaling_method standard_scale --dim_reduction_method tsne_components 10
```

## Parameters

1. __`data_period`__: Daily returns of stocks from validation period or test period (__validation__ or __test__)
    - We use validation period to choose the parameters which produces the best portfolio optimization performance.
    - Portfolio performance from test period is the true score of the proposed algorithm.
2. __`max_cluster_size`__: Maximum clustering size allowed for individual clusters (integer numbers)
3. __`scaling_method`__ : Whether scaling data to follow a normal distribution or not (__standard_scale__ or __none__)
4. __`dim_reduction_method`__ : Whether reducing dimensionality of 252-long vectors of daily returns of stocks with PCA or T-SNE or not (__PCA__, __tsne__ or __none__)
5. __`PCA_components`__ : Number of points to embed a 252-long vector using PCA. (If no value is specified, the default value 3 would be used.)
6. __`tsne_components`__ : Number of points to embed a 252-long vector using t-sne. (If no value is specified, the default value 3 would be used.)

## Datasets
- Datasets should be downloaded and preprocessed in accordance with instructions in `0. preparing_data.ipynb`, located in data folder.

## Updates as of May 16th, 2021
- 
