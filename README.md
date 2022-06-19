# ZINB (Zero Inflated Negative Binomial)


As **NumPy** array:

```python
from scipy.stats import nbinom
def ZINB(r_neg=4, p_neg=0.5, N=1000, p_zero=0.5):
    return np.random.binomial(size=N, n=1, p=(1-p_zero))*nbinom.rvs(r_neg, p_neg, size=N)
```

As **PyTorch** tensor:

```python
import torch
from scipy.stats import nbinom
def tensor_ZINB(r_neg=4, p_neg=0.5, N=1000, p_zero=0.5):
    return torch.Tensor(np.random.binomial(size=N, n=1, p=(1-p_zero))*nbinom.rvs(r_neg, p_neg, size=N))
```
