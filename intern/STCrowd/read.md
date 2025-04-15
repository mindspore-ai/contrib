STCrowd
This repository is for STCrowd dataset and mindspre implement for STCrowd: A Multimodal Dataset for Pedestrian Perception in Crowded Scenes.
[github links](https://github.com/4DVLab/STCrowd)\
[paperwithcode link](https://paperswithcode.com/paper/stcrowd-a-multimodal-dataset-for-pedestrian)
dataset file structure:
```
./
└── Path_To_STCrowd/
    ├──split.json
    ├──anno
        ├── 1.json
        ├── 2.json
        └── ...
    ├── left        
        ├── 1	
        |   ├── XXX.jpg
        |   ├── XXX.jpg
        │   └── ...
        ├── 2 
        ├── ...
    ├── right    
        ├── 1	
        |   ├── XXX.jpg
        |   ├── XXX.jpg
        │   └── ...
        ├── 2 
        ├── ...
    ├── pcd        
        ├── 1	
        |   ├── XXX.bin
        |   ├── XXX.bin
        │   └── ...
        ├── 2 
            ├── XXX.bin
            ├── XXX.bin
            └── ...
```