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

myfile="run.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter CAJ_mindspore/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../CAJ_mindspore/)

python train.py \
--MSmode "GRAPH_MODE" \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--device-id 1 \
--device-target Ascend \
--pretrain "/home/wuzesen/caj2/pretrain/resnet50.ckpt" \
--tag "sysu_all_part_ca_lrC2_Adp_NonLocalOff_New_Loss2" \
--data-path "/home/wuzesen/dataset/SYSU-MM01" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "all" \
--part 3 \
--graph False \
--epoch 90 \
--start-decay 15 \
--end-decay 90  \
--triloss "Adp"