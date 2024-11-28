# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Test Example for API."""

import argparse
from api.api_mini import TransNASBenchAPI

parser = argparse.ArgumentParser(description='TransNas-Bench-101 Dataset Path')
parser.add_argument('--data_path', type=str)
args = parser.parse_known_args()[0]

if __name__ == '__main__':
    MINDRECORD_FILE_MACRO = "macro.mindrecord"
    MINDRECORD_FILE_MICRO = "micro.mindrecord"
    MINDRECORD_FILE_INDEX = "index.mindrecord"
    API = TransNASBenchAPI(
        [args.data_path + i for i in [MINDRECORD_FILE_MICRO, MINDRECORD_FILE_MACRO, MINDRECORD_FILE_INDEX]])
    XARCH = '64-4111-basic'

    for xtask in API.task_list:
        print(f'----- {xtask} -----')
        print('--- info ---')

        for xinfo in API.info_names:
            print(f"{xinfo} : {API.get_model_info(XARCH, xtask, xinfo)}")

        print('--- metrics ---')

        for xmetric in API.metrics_dict[xtask]:

            if ('valid' in xmetric and 'loss' not in xmetric) or ('valid' in xmetric and 'neg_loss' in xmetric):
                print(f"\nbest_arch -- {xmetric}: {API.get_best_archs(xtask, xmetric, 'micro')[0]}")
