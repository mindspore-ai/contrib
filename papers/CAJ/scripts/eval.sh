#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
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

myfile="eval.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter CAJ_mindspore/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../CAJ_mindspore/)

python eval.py \
--msmode "GRAPH_MODE" \
--dataset SYSU \
--device-id 2 \
--device-target Ascend \
--resume "/home/wuzesen/caj5/logs/sysu_all_part_ca_lrC2_Ori_NonLocalOff/training/best_epoch_42_rank1_68.18_mAP_64.15_SYSU_batch-size_2*8*4=64_adam_lr_0.0035_loss-func_id+tri_P_3_main.ckpt" \
--tag "sysu_all_part_ca_lrC2_Ori_NonLocalOff" \
--data-path "/home/wuzesen/dataset/SYSU-MM01" \
--branch main \
--sysu-mode "all" \
--part 3