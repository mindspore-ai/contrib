import argparse
import mindspore
from mindspore import ops
import localquantumannealing as lqa

def main(testmode = 1):
    #set environment
    mindspore.set_context(device_target='Ascend', device_id=0)
    mindspore.set_seed(114)
    #prepare data
    couplings = ops.rand([1000,1000])
    couplings = couplings+couplings.T
    couplings = 2*couplings-1
    couplings.fill_diagonal(0.)
    if testmode:
        print("test lqa:")
        machine = lqa.Lqa(couplings)
        machine.minimise(step=2, N=1000, g=1, f=0.1)
        energy = machine.energy
        opt_time = machine.opt_time
        print("final energy:",energy)
        print("spend time",opt_time)
    else:
        print("test lqa_basic")
        machine = lqa.Lqa_basic(couplings)
        machine.minimise(step=0.1, N=1000, g=1, f=0.1, mom=0.99)
        energy = machine.energy
        opt_time = machine.opt_time
        print("final energy:",energy)
        print("spend time",opt_time)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test lqa_basic and lqa")
    parser.add_argument('--testmode', type=int, choices=[0, 1], default=1, help='Set test mode (1 for lqa, 0 for lqa_basic)')
    args = parser.parse_args()
    main(testmode=args.testmode)

