echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_train_gpu.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../train.py \
    --device_target GPU \
    --train_data_dir /path_to_cifar10/train \
    --test_data_dir /path_to_cifar10/test \
    --save_checkpoint_path /path_to_save/ \
    --log_path /path_to_log/