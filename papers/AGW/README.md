# AGW-mindspore

This is a mindspore implementation of the AGW baseline purposed in *Deep Learning for Person Re-identification:  A Survey and Outlook*. [arXiv](https://arxiv.org/abs/2001.04193v2)

## Environment

- Python 3.7
- Mindspore 1.5.0 (Ascend)

## Usage

- prepare Imagenet pretrained resnet50 checkpoints
- organize datasets as below

```sh
├──"data_path" in agw_config.yaml
   ├──market1501
      ├──Market-1501
         ├──bounding_box_train
         ├──query
         ├──bounding_box_test
   ├──dukemtmc-reid
      ├──DukeMTMC-reID
         ├──bounding_box_train
         ├──query
         ├──bounding_box_test
   ├──cuhk03
      ├──cuhk03_release
         ├──cuhk-03.mat
         ├──cuhk03_new_protocol_config_labeled.mat
         ├──cuhk03_new_protocol_config_detected.mat
   ├──msmt17
      ├──MSMT17_V1
         ├──train
         ├──test
         ├──list_val.txt
         ├──list_train.txt
         ├──list_query.txt
         ├──list_query.txt
```

- Train

```sh
python train.py --source=DATASET_NAME > $LOG_DIR 2>&1
```

- Evaluate

```sh
python eval.py --target=DATASET_NAME --checkpoint_path=TRAINED_AGW_CHECKPOINTS > $LOG_DIR 2>&1
```

where checkpoints are saved in "output_path" in config and is set to `./output` in default.
