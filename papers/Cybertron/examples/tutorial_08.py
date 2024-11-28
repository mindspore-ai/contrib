# Copyright 2020-2022 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# Contact: yangyi@szbl.ac.cn
#
# Tutorials for Cybertron
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
"""
Tutorial 08: Read hyperparameters for networ trained with force
"""

if __name__ == '__main__':

    import numpy as np
    from mindspore import dataset as ds
    from mindspore.train import Model
    from mindspore import context
    from mindspore.train import load_checkpoint

    import sys
    sys.path.append('..')

    from sponge.data import load_hyperparam
    from cybertron.cybertron import Cybertron
    from cybertron.train import MAE, RMSE
    from cybertron.train import WithForceEvalCell

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    ckpt_file = 'Tutorial_07/Tutorial_07_MolCT-best.ckpt'
    hyper_param = load_hyperparam(ckpt_file)

    net = Cybertron(hyper_param=hyper_param)
    load_checkpoint(ckpt_file, net)

    test_file = sys.path[0] + '/dataset_ethanol_origin_testset_1024.npz'
    test_data = np.load(test_file)

    scale = test_data['scale']
    shift = test_data['shift']

    net.set_scaleshift(scale=scale, shift=shift, unit='kj/mol')

    net.print_info()

    ds_test = ds.NumpySlicesDataset(
        {'R': test_data['R'], 'F': test_data['F'], 'E': test_data['E']}, shuffle=False)
    ds_test = ds_test.batch(1024)
    ds_test = ds_test.repeat(1)

    eval_network = WithForceEvalCell('RFE', net)

    outdir = 'Tutorial_08'
    outname = outdir + '_' + net.model_name

    energy_mae = 'EnergyMAE'
    forces_mae = 'ForcesMAE'
    forces_rmse = 'ForcesRMSE'
    model = Model(net, eval_network=eval_network,
                  metrics={energy_mae: MAE([1, 2]), forces_mae: MAE([3, 4]),
                           forces_rmse: RMSE([3, 4], atom_aggregate='sum')})

    print('Test dataset:')
    eval_metrics = model.eval(ds_test, dataset_sink_mode=False)
    info = ''
    for k, v in eval_metrics.items():
        info += k
        info += ': '
        info += str(v)
        info += ', '
    print(info)

    outdir = 'tutorial_08'
    scaled_ckpt = outdir + '_' + net.model_name + '.ckpt'
    net.save_checkpoint(scaled_ckpt, outdir)

    net2 = Cybertron(hyper_param=load_hyperparam(outdir+'/'+scaled_ckpt))
    net2.print_info()
