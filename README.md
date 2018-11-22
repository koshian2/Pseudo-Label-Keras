## Pseudo-Label-Keras
A semi-supervised learning approach using pseudo labels. Implemented on Keras, evaluated on CIFAR-10.

## Result
### VGG-like 10-layer CNN
| # True Labels | Supervised | Pseudo Labels | Pseudo Gain | # Pseudo Labels |
|:-------------:|-----------:|--------------:|------------:|----------------:|
|      500      |     39.81% |        39.02% |      -0.79% |           49500 |
|      1000     |     47.95% |        50.89% |       2.94% |           49000 |
|      5000     |     69.41% |        74.58% |       5.17% |           45000 |
|     10000     |     77.96% |        80.46% |       2.50% |           40000 |

### vs MobileNet Transfer Leaning 
| Pseudo Labels | ImageNet Weights |  N=500 | N=1000 | N=5000 | N=10000 |
|:-------------:|:----------------:|-------:|-------:|-------:|--------:|
|       No      |        Yes       | 51.77% | 60.13% | 68.23% |  74.92% |
|       No      |        No        | 22.70% | 30.81% | 55.27% |  65.77% |
|      Yes      |        No        | 34.26% | 44.95% | 63.39% |  72.94% |
|      Yes      |        Yes       | 46.14% | 51.37% | 65.00% |  73.86% |

## Reference
Dong-Hyun, Lee. Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. 2013
[http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf)

## Details(Japanese)
CIFAR-10を疑似ラベル（Pseudo-Label）を使った半教師あり学習で分類する
https://qiita.com/koshian2/items/f4a458466b15bb91c7cb
