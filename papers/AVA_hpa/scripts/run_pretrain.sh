echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_pretrain.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../pretrain.py \
    --device_id 0 \
    --device_num 1 \
    --device_target Ascend \
    --data_dir /path_to_hpa \
    --save_checkpoint_path /path_to_save_ckpt/ \
    --log_path /path_to_log/