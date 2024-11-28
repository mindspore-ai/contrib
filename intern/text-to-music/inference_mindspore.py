import os
import time
import mindspore as ms
import random
import argparse
from unidecode import unidecode
from samplings import top_p_sampling, temperature_sampling
from mindnlp.transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_abc(args):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    num_tunes = args.num_tunes
    max_length = args.max_length
    top_p = args.top_p
    temperature = args.temperature
    seed = args.seed
    print(" HYPERPARAMETERS ".center(60, "#"), '\n')
    args = vars(args)
    for key in args.keys():
        print(key+': '+str(args[key]))

    with open('input_text.txt') as f:
        text = unidecode(f.read())
    print("\n"+" INPUT TEXT ".center(60, "#"))
    print('\n'+text+'\n')

    tokenizer = AutoTokenizer.from_pretrained('sander-wood/text-to-music',clean_up_tokenization_spaces=True)
    model = AutoModelForSeq2SeqLM.from_pretrained('sander-wood/text-to-music')
    #model = model.to(device)

    input_ids = tokenizer(text,
                          return_tensors='ms',
                          truncation=True,
                          max_length=max_length)['input_ids']
    decoder_start_token_id = model.config.decoder_start_token_id
    eos_token_id = model.config.eos_token_id
    random.seed(seed)
    tunes = ""
    print(" OUTPUT TUNES ".center(60, "#"))

    for n_idx in range(num_tunes):
        print("\nX:"+str(n_idx+1)+"\n", end="")
        tunes += "X:"+str(n_idx+1)+"\n"
        decoder_input_ids = ms.tensor([[decoder_start_token_id]])

        for t_idx in range(max_length):

            if seed != None:
                n_seed = random.randint(0, 1000000)
                random.seed(n_seed)
            else:
                n_seed = None
            outputs = model(input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids)
            probs = outputs.logits[0][-1]
            #probs = ms.nn.Softmax(dim=-1)(probs).cpu().detach().numpy()
            softmax=ms.nn.Softmax()
            probs = softmax(probs).numpy()
            sampled_id = temperature_sampling(probs=top_p_sampling(probs,
                                                                   top_p=top_p,
                                                                   seed=n_seed,
                                                                   return_probs=True),
                                              seed=n_seed,
                                              temperature=temperature)
            decoder_input_ids = ms.ops.cat(
                (decoder_input_ids, ms.tensor([[sampled_id]])), 1)
            if sampled_id != eos_token_id:
                sampled_token = tokenizer.decode([sampled_id])
                print(sampled_token, end="")
                tunes += sampled_token
            else:
                tunes += '\n'
                break

    timestamp = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
    with open('output_tunes/'+timestamp+'.abc', 'w') as f:
        f.write(unidecode(tunes))


def get_args(parser):

    parser.add_argument('-num_tunes', type=int, default=3,
                        help='the number of independently computed returned tunes')
    parser.add_argument('-max_length', type=int, default=1024,
                        help='integer to define the maximum length in tokens of each tune')
    parser.add_argument('-top_p', type=float, default=0.9,
                        help='float to define the tokens that are within the sample operation of text generation')
    parser.add_argument('-temperature', type=float, default=1.,
                        help='the temperature of the sampling operation')
    parser.add_argument('-seed', type=int, default=None,
                        help='seed for randomstate')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    generate_abc(args)
