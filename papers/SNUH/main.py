"""Main modules."""
from model.SNUH import SNUH

if __name__ == "__main__":
    argparser = SNUH.get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = SNUH(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))
