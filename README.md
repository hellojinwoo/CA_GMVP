# Clustering Approaches for GMVP

This is the source code used for experiments for the research paper "<a href = "https://arxiv.org/abs/2001.02966">__Clustering Approaches for Global Minimum Variance Portfolio__"</a>

## Example
```python
python3 main.py --data_period 'test' --max_cluster_size 75 --scaling_method 'none' --dim_reduction_method 'none'
```

**Parameters**

- __`data_period`__: Daily returns of stocks from validation period or test period (`validation` or `test`)
  - We use validation period to choose the parameters which produces the best portfolio optimization performance.
  - Portfolio performance from test period is the true score of the proposed algorithm.
- __`max_cluster_size`__: Maximum clustering size allowed for individual clusters (integer numbers)
- __
  (or tensor) of shape `[num_seqs, seq_len, num_features]` representing your training set of sequences.
  - Each sequence should have the same length, `seq_len`, and contain a sequence of vectors of size `num_features`.
  - If `num_features=1`, then you can input a list of shape `[num_seqs, seq_len]` instead.
  - __[Notice]__ Currently TorchCoder can take `[num_seqs, seq_len]` as an input. Soon to be fixed.
- `embedding_dim`: Size of the vector encodings you want to create.
- `learning_rate`: Learning rate for the autoencoder. default = 1e-3
- `every_epoch_print` : Deciding the size of N to print the loss every N epochs. default = 100
- `epochs`: Total number of epochs to train for. default = 10000
- `patience` : Number of epochs to wait for if the loss does not decrease. default = 20 
- `max_grad_norm` : Maximum size for gradient used in gradient descent (gradient clipping). default = 0.005

### Datasets
Datasets should be downloaded and preprocessed according to instructions in `0.preparing_data.ipynb`.

The introduction page would be updated with more details in the foreseeabale future. (Last updated in Jan 10, 2020)
