#!/bin/bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_classifier_gpu.sh DEVICE_ID"
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_classifier_gpu.sh DEVICE_ID 1"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

if [ -z $1 ]
then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES="$1"
fi

mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export MS_DEV_ENABLE_FALLBACK=0
echo "start"
python ${PROJECT_DIR}/../run_classifier.py  \
    --config_path="../../task_classifier_config.yaml" \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="false" \
    --assessment_method="Accuracy" \
    --epoch_num=5 \
    --num_class=2 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=8 \
    --eval_batch_size=64 \
    --save_finetune_checkpoint_path="____model____.ckpt" \
    --load_pretrain_checkpoint_path_bert="bert_base_ascend_v130_zhwiki_official_nlp_bs256_acc91.72_recall95.06_F1score93.36.ckpt" \
    --load_pretrain_checkpoint_path_vit="vit_b_16_224.ckpt" \
    --load_finetune_checkpoint_path="/root/huawei/____model____.ckpt/classifier-10_597.ckpt" \
    --train_data_file_path="/root/TextCaps-OCR" \
    --eval_data_file_path="/root/TextCaps-OCR"