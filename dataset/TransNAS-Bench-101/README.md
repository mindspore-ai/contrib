# TransNAS-Bench-101: Improving Transferrability and Generalizability of Cross-Task Neural Architecture Search

We propose TransNAS-Bench-101, a benchmark containing network performance across seven tasks, covering classification, regression, pixel-level prediction, and selfsupervised tasks. This diversity provides opportunities to transfer NAS methods among the tasks and allows for more complex transfer schemes to evolve. We explore two fundamentally different types of search spaces: cell-level search space and macro-level search space. With 7,352 backbones evaluated on seven tasks, 51,464 trained models with detailed training information are provided. With TransNASBench-101, we hope to encourage the advent of exceptional NAS algorithms that raise cross-task search efficiency and generalizability to the next level.

In this Markdown file, we show an example how to use TransNAS-Bench-101

## How to use TransNAS-Bench-101

1. Import the API object in `./api/api_mini.py` and create an API instance from the `.mindrecord` file in the [dataset](https://download.mindspore.cn/dataset/TransNAS-Bench-101/):

    ```python
    from api.api_mini import TransNASBenchAPI
    path2nas_bench_files_list = ["macro.mindrecord", "micro.mindrecord", "index.mindrecord"]
    API = TransNASBenchAPI(path2nas_bench_files_list)
    ```

2. Check the task information, number of architectures evaluated, and search spaces:

    ```python
    # show number of architectures and number of tasks
    length = len(API)
    task_list = API.task_list # list of tasks
    print(f"This API contains {length} architectures in total across {len(task_list)} tasks.")
    # This API contains 7352 architectures in total across 7 tasks.

    # Check all model encoding
    search_spaces = API.search_spaces # list of search space names
    all_arch_dict = API.all_arch_dict # {search_space : list_of_architecture_names}
    for ss in search_spaces:
      print(f"Search space '{ss}' contains {len(all_arch_dict[ss])} architectures.")
    print(f"Names of 7 tasks: {task_list}")
    # Search space 'macro' contains 3256 architectures.
    # Search space 'micro' contains 4096 architectures.
    # Names of 7 tasks: ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
    ```

3. Since different tasks may require different evaluation metrics, hence `metric_dict` showing the used metrics can be retrieved from `API.metrics_dict`. TransNAS-Bench API also recorded the model inference time, backbone/model parameters, backbone/model FLOPs in `API.infor_names`.

    ```python
    metrics_dict = API.metrics_dict # {task_name : list_of_metrics}
    info_names = API.info_names # list of model info names

    # check the training information of the example task
    task = "class_object"
    print(f"Task {task} recorded the following metrics: {metrics_dict[task]}")
    print(f"The following model information are also recorded: {info_names}")
    # Task class_object recorded the following metrics: ['train_top1', 'train_top5', 'train_loss', 'valid_top1', 'valid_top5', 'valid_loss', 'test_top1', 'test_top5', 'test_loss', 'time_elapsed']
    # The following model information are also recorded: ['inference_time', 'encoder_params', 'model_params', 'model_FLOPs', 'encoder_FLOPs']
    ```

4. Query the results of an architecture by arch string

    ```python
    # Given arch string
    xarch = API.index2arch(1) # '64-2311-basic'
    for xtask in API.task_list:
        print(f'----- {xtask} -----')
        print(f'--- info ---')
        for xinfo in API.info_names:
            print(f"{xinfo} : {API.get_model_info(xarch, xtask, xinfo)}")
        print(f'--- metrics ---')
        for xmetric in API.metrics_dict[xtask]:
            print(f"{xmetric} : {API.get_single_metric(xarch, xtask, xmetric, mode='best')}")
            print(f"best epoch : {API.get_best_epoch_status(xarch, xtask, metric=xmetric)}")
            print(f"final epoch : {API.get_epoch_status(xarch, xtask, epoch=-1)}")
            if ('valid' in xmetric and 'loss' not in xmetric) or ('valid' in xmetric and 'neg_loss' in xmetric):
                print(f"\nbest_arch -- {xmetric}: {API.get_best_archs(xtask, xmetric, 'micro')[0]}")
    ```

A complete example is given in `example.py`

- `python example.py --data_path [data_path]`

## Example network encoding in both search spaces

```Macro example network: 64-1234-basic
- Base channel: 64
- Macro skeleton: 1234 (4 stacked modules)
  - [m1(normal)-m2(channelx2)-m3(resolution/2)-m4(channelx2 & resolution/2)]
- Cell structure: basic (ResNet Basic Block)

Micro example network: 64-41414-1_02_333
- Base channel: 64
- Macro skeleton: 41414 (5 stacked modules)
  - [m1(channelx2 & resolution/2)-m2(normal)-m3(channelx2 & resolution/2)-m4(normal)-m5(channelx2 & resolution/2)]
- Cell structure: 1_02_333 (4 nodes, 6 edges)
  - node0: input tensor
  - node1: Skip-Connect( node0 ) # 1
  - node2: None( node0 ) + Conv1x1( node1 ) # 2
  - node3: Conv3x3( node0 ) + Conv3x3( node1 ) + Conv3x3( node2 ) # 3
```

## Citation

If you find that TransNAS-Bench-101 helps your research, please consider citing it:
