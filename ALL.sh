### FMNIST
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=0 python main.py --dataset fmnist --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=1 python main.py --dataset fmnist --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=2 python main.py --dataset fmnist --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

### CIFAR10
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

### CINIC10
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cinic10 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

### CIFAR100
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.01 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.04 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32

CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 8 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 6 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --block_dim BC --use_quantization --quantization_bits 4 --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --batch_size 32
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --alpha 0.16 --model_name ConvNet --c_rounds 200 --project_name FL_experiment --moving_average --ma_start 1 --moving_weight 0.9 --batch_size 32
