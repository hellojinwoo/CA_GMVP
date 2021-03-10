# Clustering Approaches for GMVP

This is the source code used for experiments for the research paper "<a href = "https://arxiv.org/abs/2001.02966">__Clustering Approaches for Global Minimum Variance Portfolio__"</a>

## Example
```
python3 main.py --data_period 'test' --max_cluster_size 75 --scaling_method 'none' --dim_reduction_method 'none'
```

## Parameters

- __`data_period`__: Daily returns of stocks from validation period or test period (__`validation`__ or __`test`__)
  - We use validation period to choose the parameters which produces the best portfolio optimization performance.
  - Portfolio performance from test period is the true score of the proposed algorithm.
- __`max_cluster_size`__: Maximum clustering size allowed for individual clusters (integer numbers)
- __`scaling_method`__ : Whether scaling data to follow a normal distribution or not (__`standard_scale`__ or __`none`__)
- __`dim_reduction_method`__ : Whether reducing dimensionality of 252-long vectors of daily returns of stocks with PCA or T-SNE or not (__`PCA`__, __`tsne`__ or __`none`__)

## Datasets
Datasets should be downloaded and preprocessed in accordance with instructions in `0. preparing_data.ipynb`, located in data folder.
