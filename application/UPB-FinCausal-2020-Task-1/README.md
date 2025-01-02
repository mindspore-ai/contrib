# UPB FinCausal 2020 Task 1

**Instructions：** I migrated the Pytorch code from the original repository to Mindspore

This repository contains the ensemble used by the UPB team to obtain the 2nd place at the [FinCausal 2020](http://wp.lancs.ac.uk/cfie/fincausal2020/) Task 1. The ensemble is composed of five Transformer-based models: Bert-Large, RoBERTa-Large, ALBERT-Base, FinBERT-Base and SciBERT-base. The training of the models was done using the [HuggingFace](https://github.com/huggingface/transformers) library.

<p align="center">
  <img src="https://raw.githubusercontent.com/avramandrei/UPB-FinCausal-2020-Task-1/main/resources/Ensemble-Figure.png">
</p>

The ensemble can be downloaded from [here](https://swarm.cs.pub.ro/~ccercel/UPB-Fincausal2020-best-ensemble.zip).

## Installation

Make sure you have Python3 and MindSpore installed. Then, install the dependencies via pip:

```
pip install -r requirements.txt
```

## Prediction

To make a prediction run the `predict.py` script and give it an input file with a sentence on each line and the path to the ensemble of models.

```
python3 predict.py [input_path] [ensemble_path] [--output_path]
```

The script will output a file with 1 or 0 on each line, for each model and for the ensemble, corresponding to the sentence being causal or not being causal, respectively. 

```
albert-base     bert-large      finbert-base    roberta-large   scibert-base    ensemble        
0               0               0               0               0               0               
0               0               0               0               0               0               
1               1               1               1               1               1               
1               1               1               1               1               1               
1               1               1               1               1               1               
```

## Ensemble Performance

We depict the performance of each individual model and of the ensemble on both validation dataset (split explained in paper) and on the evaluation dataset.

| Model | Valid-Prec | Valid-Rec | Valid-F1c | Test-Prec | Test-Rec | Test-F1 |
--------| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
BERT-Large | 95.81 | 96.15 | 95.77 | 97.10 | 97.02 | 97.05 |
RoBERTa-Large | 95.69 | 95.29 | 95.46 | 97.35 | 97.30 | 97.32 |
ALBERT-Base | 95.71 | 95.88 | 95.78 | 96.75 | 96.76 | 96.75 | 
FinBERT-Base | 93.88 | 94.71 | 93.92 | 94.08 | 94.30 | 94.18 |
SciBERT-Base | 95.67 | 95.99 | 95.75 | 96.77 | 96.83 | 96.89 | 
Ensemble | - | - | - | **97.53** | **97.59** | **97.55** |

## Cite
If you are using this repository, please consider citing the following [paper](https://www.aclweb.org/anthology/2020.fnp-1.8.pdf) as a thank you to the authors: 
```
@inproceedings{ionescu-etal-2020-upb,
    title = "{UPB} at {F}in{C}ausal-2020, Tasks 1 {\&} 2: Causality Analysis in Financial Documents using Pretrained Language Models",
    author = "Ionescu, Marius  and
      Avram, Andrei-Marius  and
      Dima, George-Andrei  and
      Cercel, Dumitru-Clementin  and
      Dascalu, Mihai",
    booktitle = "Proceedings of the 1st Joint Workshop on Financial Narrative Processing and MultiLing Financial Summarisation",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "COLING",
    url = "https://www.aclweb.org/anthology/2020.fnp-1.8",
    pages = "55--59",
    abstract = "Financial causality detection is centered on identifying connections between different assets from financial news in order to improve trading strategies. FinCausal 2020 - Causality Identification in Financial Documents {--} is a competition targeting to boost results in financial causality by obtaining an explanation of how different individual events or chain of events interact and generate subsequent events in a financial environment. The competition is divided into two tasks: (a) a binary classification task for determining whether sentences are causal or not, and (b) a sequence labeling task aimed at identifying elements related to cause and effect. Various Transformer-based language models were fine-tuned for the first task and we obtained the second place in the competition with an F1-score of 97.55{\%} using an ensemble of five such language models. Subsequently, a BERT model was fine-tuned for the second task and a Conditional Random Field model was used on top of the generated language features; the system managed to identify the cause and effect relationships with an F1-score of 73.10{\%}. We open-sourced the code and made it available at: https://github.com/avramandrei/FinCausal2020.",
}
```



