---
Selected-Topics-in-Visual-Recognition-using-Deep-Learning HW2
---
# Street View House Numbers
[Reproducing the work](#Reproducingthework)  
[Enviroment Installation](#Enviroment Installation)
[Project installation](#Project installation)
[Training](#Training)  
[Inference](#Inference) 
[Visualization](#Visualization)

## Reproducing the work
### Enviroment Installation
1. install annoconda
2. create python3.x version 
    ```
    take python3.6 for example
    $ conda create --name (your_env_name) python=3.6
    $ conda activate (your_env_name)
    ```
3. install pytorch ([check GPU version](https://www.nvidia.com/Download/index.aspx?lang=cn%20))
    - [pytorch](https://pytorch.org/get-started/locally/)
### Project installation
1. clone this repository
    ``` 
    git clone https://github.com/q890003/HW2_Street_View_House_Numbers.git
    ```
2. Data
    1. Download Official Image: 
        - [Test data](https://drive.google.com/file/d/1vvnqdtFzze_YESyjE6_XOPO0ZML4auEE/view?usp=sharing)
        - [Train data](https://drive.google.com/file/d/1kEGY_vVCw_iUrbSquKplLp0PeoXq5PQq/view?usp=sharing)

    2. Put (Test/Train) data to folder, **data/**, under the root dir of this project. 
        ```
        |- HW2_Street_View_House_Numbers
            |- data/
                |- dummy.pkl 
                |- test.zip
                |- train.tar.gz
            |- checkpoints/   (auto-gen by train.py)
                |- (step3. parameter_file of model)
            |- results/       (auto-gen by train.py)
            |- datasets/
            |- .README.md
            |- train.py
            |- eval.py
            |- bbox_visualization.py
            |- config.py
            ...
        ```
    3. Decompress the (Test/Train) data
        ```
        At dir HW2_Street_View_House_Numbers/data/
        $ unzip ./test.zip 
        $ tar zxvf ./train.tar.gz
        ```
4. Downoad fine-tuned parameters
    - [resnet101_parameters](https://drive.google.com/file/d/1lYDxtcELuzWSlnekhOOjHC04EHt55SM2/view?usp=sharing)
    - put the parameter file to checkpoints folder.
## Training
```
$ python train.py
``` 
## Inference

```
$ python eval.py
```
## Visualization
```
Note: Change path to the json file of prediction result
$ python bbox_visualization.py
```
