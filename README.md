
# Modality-Aware Representation Learning for Zero-shot Sketch-based Image Retrieval

This is an implementation of the paper "Modality-Aware Representation Learning for Zero-shot Sketch-based Image Retrieval", accepted at WACV 2024.

- this includes **categorical ZS-SBIR**
- this does not include **instance-level ZS-SBIR**

This repository has been updated to reflect the version used for our submission. (Keep in mind that there might be slight differences since we did our experiments with another private repository.)

You can find the **pretrained model weights** at this [link](https://drive.google.com/drive/folders/1R9YaEDMOv5sap_ohgPlo6JYxPXkgS0eQ?usp=sharing).


## setup
- prepare a virtual environment

        conda create --name sbir python=3.8

- install the required packages

        pip install -r requirements.txt

- download necessary datasets
    - quickdraw-extended
    - sketchy-extended
    - tuberlin-extended

- update lines 12 to 16 in [get_loader.py](data/get_loader.py) with the paths where you downloaded each dataset

    ```python
    # example
    data_dir_paths = [
        "/data/dataset/tuberlin-ext/",
        "/data/dataset/QuickDraw-Extended/",
        "/data/dataset/sketchy/Sketchy",
    ]
    ```


## training
- update setting file ([config-train.yaml](setting/config-train.yaml))
    - Our code distinguishes between different datasets using indices.

        |Idx|dataset|
        |------|---|
        |0|TU-berlin|
        |1|QuickDraw|
        |2|Sketchy|

- run [train.py](train.py)

        python train.py

- after successful training, new output folders(`outputs`, `wandb`) are generated as shown below

    ```
        .
        ├── data
        ├── model
        ├── outputs/
        │   └── 2024-05-10/
        │       └── 22-31-19/
        │           ├── .hydra
        │           ├── model
        │           └── train.log
        ├── setting
        ├── wandb/
        │   └── ...(omitted)
        ├── .gitignore
        └── ...(omitted)
    ```


## inference
- update setting file ([config-infer.yaml](setting/config-infer.yaml))
- run [infer.py](infer.py)

        python infer.py
    
- this will use trained model weights to generate logits, creating `infer_output` folder.
    ```
        .
        ├── ...(omitted)
        ├── outputs/
        │   └── 2024-05-10/
        │       └── 22-31-19/
        │           ├── .hydra
        │           ├── model
        │           ├── infer_output/
        │           │   └── best/
        │           │       └── 0/ <--(dataset index)
        │           │           ├── converted_logits.pt
        │           │           ├── img_logits.pt
        │           │           └── txt_logits.pt
        │           └── train.log
        └── ...(omitted)
    ```

## evaluation
- update setting file ([config-eval.yaml](setting/config-eval.yaml))
- run [evaluate.py](evaluate.py)

        python evaluate.py

- this process shows metric results and the top 10 retrieved photos for each sketch image. you can check these in the newly generated file (named `eval_result.json`), after running code


## citing
```
@InProceedings{Lyou_2024_WACV,
    author    = {Lyou, Eunyi and Lee, Doyeon and Kim, Jooeun and Lee, Joonseok},
    title     = {Modality-Aware Representation Learning for Zero-Shot Sketch-Based Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5646-5655}
}
```