"""
Filename: communication_gan.py
Author: fangxiuwen
Contact: fangxiuwen67@163.com
"""
import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--private_classes', type=list, default=[10, 11, 12, 13, 14, 15], help='private_classes_list')
    parser.add_argument('--public_classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='public_classes_list')
    parser.add_argument('--N_parties', type=int, default=10, help='the number of parties')
    parser.add_argument('--N_samples_per_class', type=int, default=12, help='N_samples_per_class')
    parser.add_argument('--gpu', default='0', type=str, help='set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--parameter-server', default=False)
    parser.add_argument('--run-distribute', action='store_true',
                        help="if set true, this code will be run on distributed architecture with mindspore")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--output_classes', type=int, default=16, help='set the output points')
    parser.add_argument('--user_number', type=int, default=10, help='number of user join in Federated Learning')
    parser.add_argument('--collaborative_epoch', type=int, default=1,
                        help='collaborative_epoch for train on public mnist')
    parser.add_argument('--Communicationepoch', type=int, default=1, help='Collaobrative epoch in Step3') #30
    parser.add_argument('--Communication_private_epoch',type=int, default=2,
                        help='Local private training during colaboratiive time')
    parser.add_argument('--Communication_domain_identifier_epochs', type=int, default=4,
                        help='gan domain identitfer growup')
    parser.add_argument('--Communication_gan_local_epochs', type=int, default=4, help='gan domain identitfer growup')
    args = parser.parse_args(args=[])
    return args
