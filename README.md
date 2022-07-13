# Clustering Approaches for Global Mininum Variance Portfolio

- This is the source code used for experiments for the research paper "<a href = "https://arxiv.org/abs/2001.02966">__Clustering Approaches for Global Minimum Variance Portfolio__"</a>
- The academic paper utilizes 'constrained K-means clustering' to group stocks showing similar price movements before performing 'within cluster portfolio optimization'. 
<p align="center">
<img src="https://user-images.githubusercontent.com/34431729/118437017-1ff5d200-b6e2-11eb-848d-a7f99d0e0019.png" width="800">
</p>

## Example
- Using raw data without scaling methods and dimensional reduction methods
```
python main.py --data_period test --max_cluster_size 75 --scaling_method none --dim_reduction_method none
```
- Using PCA without scaling methods (If `PCA_components` is not specfied, the default number 3 is used) 
```
python main.py --data_period test --max_cluster_size 75 --scaling_method none --dim_reduction_method PCA
```
- Using t-sne with standard scaling and t-sne components = 10
```
python main.py --data_period test --max_cluster_size 75 --scaling_method standard_scale --dim_reduction_method tsne --tsne_components 10
```

## Hyper-Parameters

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
1. Codes are fixed and improved to prevent errors. (For example, global variables are not used anymore.)
2. Number of `PCA_components` and `tsne_components` can be provided using argparse, which makes it easier to use dimensionality reduction methods.
3. The library `cudf` is now replaced with `sklearn`, due to more ease of use. 
