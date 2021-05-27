echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_train.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../train.py \
    --device_id 0 \
    --device_num 1 \
    --device_target Ascend \
    --train_data_dir /path_to_cifar10/train \
    --test_data_dir /path_to_cifar10/test \
    --save_checkpoint_path /path_to_save/ \
    --log_path /path_to_log/