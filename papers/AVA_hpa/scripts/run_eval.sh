echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_eval.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../eval.py \
    --device_id 0 \
    --device_num 1 \
    --device_target Ascend \
    --model_arch resnet18 \
    --classes 10 \
    --ckpt_path /path_to_load_ckpt_for_eval/ \
    --data_dir /path_to_hpa/ \
    --save_eval_path /path_to_save_eval_result/