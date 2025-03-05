# [NeurIPS 2024] Low Precision Local Training is Enough for Federated Learning

This repository contains a PyTorch implementation of the paper:

[Low Precision Local Training is Enough for Federated Learning.](https://openreview.net/pdf?id=vvpewjtnvm)

Zhiwei, Li and Yiqiu, Li and Binbin, Lin and Zhongming, Jin and Weizhong, Zhang

## Strength
1. **Better performance and lower cost**

![acc vs cost](assets/lowp.png)

$~~~~~~~$ Our method outperforms other efficient Federated Learning (FL) methods on accuracy and traning & communication cost. We take [HeteroFL](https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients) and [SplitMix](https://github.com/illidanlab/SplitMix) for example, as both of them contribute to decrease training & communication cost during FL.

2. **Compatibility with multiple FL methods**

![fedavg](assets/fedavg.png)

![otherfl](assets/otherfl.png)

$~~~~~~~$ Our method can maintain, and sometimes even surpass, the accuracy of standard FL approaches. Moreover, it is compatible with various FL methods, as shown in the table below.

|Normal|Regularization-based|Data-dependent knowledge distill|Data-free knowledge distill|
|:-:|:-:|:-:|:-:|
|[FedAVG](https://arxiv.org/pdf/1602.05629)|[FedProx](https://arxiv.org/pdf/1812.06127)|[ABAvg](https://ieeexplore.ieee.org/document/9521631)|[FedFTG](https://arxiv.org/pdf/2203.09249), [FedGen](https://arxiv.org/pdf/2105.10056)|

3. **Reduction of local training overfitting**

![overfitting](assets/leov.png)

$~~~~~~~$ Due to the quantization noise introduced during local training, the overfitting problem is alleviated.

## Introduction & Methods
In this paper, we propose an efficient FL paradigm that significantly reduces the communication and computation costs during training. The key features of our approach are:

1. **Low-Precision Local Training**: The local models at clients are trained using low-precision operations (as low as 8 bits), which reduces both the computational load and memory usage.

$$w^{k} _{t+1} = Q (w^{k} _{t} - \eta _t \nabla F_k(w^{k} _{t}; \xi^k _t))$$

2. **Low-Precision Communication**: The client and server send low-precision model weights to each other, reducing communication overhead.

$$w^{k} _{t} \leftarrow Q (\bar{w}_t)$$

3. **High-Precision Aggregation**: Only the model aggregation in the server is performed using high-precision computation, ensuring that the final model accuracy is preserved.

$$w _{t+E} = \sum _{k\in S_t} \frac{p_k}{q_t} w^{k} _{t+E}$$

$~~~~~~~$ To metigate the quantization error, we maintain a moving average in the server.

$$\bar{w} _{t} = \lambda \bar{w} _{t-E}+(1-\lambda)w_t$$

Our experimental results show that models trained with 8-bit precision perform comparably to those trained with full precision, demonstrating the effectiveness of our approach in maintaining high performance while significantly reducing resource consumption.

## Code

Our implementation has minimal dependencies. Additionally, the GPUs used in the experiments are the GTX 4090 and A6000. To get started with the implementation of our method, you can clone the repository and follow the instructions below.

```bash
# Clone the repository
git clone https://github.com/digbangbang/LPT-FL.git

# Install dependencies
pip install -r requirements.txt

# Run the demo script
python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
```

The whole implementation of **FedAvg** are in ALL.sh, other FL methods are as followings.

**FedProx** experiments:
```
python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --aggregation_mode fedprox --lambda_fedprox 0.1 --project_name FL_experiment_FedProx --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
```

**ABAvg** experiments:
```
python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --aggregation_mode abavg --project_name FL_experiment_abavg --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
```

**FedTGP** experiments:
```
python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --aggregation_mode fedtgp --project_name FL_experiment_FedTGP --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32 --lr 1e-3 --lr_schedule CosineAnnealingLR
```

**FedGen** experiments:
```
python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --aggregation_mode fedgen --project_name FL_experiment_Fedgen --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32 --lr 1e-4 --lr_schedule CosineAnnealingLR
```

## Results

![results](assets/fedavgout.png)

The performance of FedAvg with 8 bit local training surpasses the original version on 4 datasets.

## Acknowledgements
This project uses modified code from the following projects:

- [SWALP](https://github.com/stevenygd/SWALP): developed by Cornell-CS. Block floating point quantization codes reused for low precision training. See models/quantizer.py.

## Cite

If you find our paper useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{li2025low,
  title={Low Precision Local Training is Enough for Federated Learning},
  author={Li, Zhiwei and Li, Yiqiu and Lin, Binbin and Jin, Zhongming and Zhang, Weizhong},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={90160--90197},
  year={2025}
}
```
