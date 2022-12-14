# From Within to Between: Knowledge Distillation for Cross Modality Retrieval

### Dependencies
The code assumes PyTorch 1.4 and Python 3.7 (other versions may work, but have not been tested).

Dependencies can be installed via:

```pip install -r requirements/pip-requirements.txt```


### Model Zoo

Please note that the numbers are slightly different in the paper due to compression artifacts correction.

**MSRVTT**

| Model | Distillation | Task | R@1 | R@5 | R@10 | R@50 | MdR | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | ----- |
| CE+    | Caption  | t2v  | 14.5 | 37.7 | 50.6 | 78.5 | 10.0 |  [config](configs_crosskd/msrvtt-train-gpt2-xl-finetuned-mte-adam-cecap.json), [model](https://drive.google.com/file/d/1KOUPBLQlhqmtOGBSIjLIUXHfOAppj8Y5/) |
| TeachText - CE+    | Caption  | t2v  | 14.8 | 38.1 | 51.1 | 79.1 | 10.0  | [config](configs_crosskd/msrvtt-train-gpt2-xl-finetuned-mte-adam-ttcap.json), [model](https://drive.google.com/file/d/14vjfwyZi9qYTJKMEwCv6z_tHm5eqD5ih/) |

**MSVD**

| Model | Distillation | Task | R@1 | R@5 | R@10 | R@50 | MdR | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | 
| CE+    | Caption  | t2v  | 25.9 | 58.3 | 72.7 | 93.3 | 4.0 | [config](configs_crosskd/msvd-train-gpt2-xl-finetuned-mte-adam-cecap.json), [model](https://drive.google.com/file/d/1atl8S1d-yFjnt_M4gu602MZcqH5FEWqd/) |
| TeachText - CE+    | Caption  | t2v  | 25.6 | 57.1 | 71.4 | 92.9 | 4.0 | [config](configs_crosskd/msvd-train-gpt2-xl-finetuned-mte-adam-ttcap.json), [model](https://drive.google.com/file/d/1D4btmyYIHW-AQLe3HBODBcAYkbBXMrEG/) |

**DiDeMo**

| Model | Distillation | Task | R@1 | R@5 | R@10 | R@50 | MdR | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | ----- |
| CE+    | Caption  | t2v  | 19.5 | 44.6 | 59.0 | 83.0 | 7.0 | [config](configs_crosskd/didemo-train-gpt2-xl-finetuned-mte-adam-cecap.json), [model](https://drive.google.com/file/d/1PP3C_aHMeL0pkAISE5LKFcEqpFjFWFNF) |
| TeachText - CE+    | Caption  | t2v  | 22.2 | 49.0 | 61.4 | 86.4 | 6.0 | [config](configs_crosskd/didemo-train-gpt2-xl-finetuned-mte-adam-ttcap.json), [model](https://drive.google.com/file/d/1d4Y3vWw7vcOXjAjvxAdj8s45fs_lidOO/) |

**Activity-Net**

| Model | Distillation | Task | R@1 | R@5 | R@10 | R@50 | MdR | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | ----- |
| CE+    | Caption  | t2v  | 20.0 | 50.8 | 67.3 | 93.7 | 5.0 | [config](configs_crosskd/activity-net-train-gpt2-xl-finetuned-mte-adam-cecap.json), [model](https://drive.google.com/file/d/1EbXPBhHPWBxluGbPZxss4LgKVL-EMaq-) |
| TeachText - CE+    | Caption  | t2v  | 23.8 | 56.9 | 73.2 | 96.2 | 4.0 | [config](configs_crosskd/activity-net-train-gpt2-xl-finetuned-mte-adam-ttcap.json), [model](https://drive.google.com/file/d/1SbB6eEa341us17f7AGpMQpjw4K9Fbm42) |

### Data for training
You can download the high quality features used for TeachText from:

```
For MSRVTT:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-MSRVTT-experts.tar.gz
sha1sum: 734650c3b98509996da75cdedc12101836624917

For MSVD:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-MSVD-experts.tar.gz
sha1sum: c8eba8c5291dd6bb501757ed0cc327cd22217965

For DiDeMo:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-DiDeMo-experts.tar.gz
sha1sum: 8e128309f12cf3260fe538f82578b5ad91a46bd0

For ActivityNet:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-activity-net-experts.tar.gz
sha1sum: 2f3c7c2fe86bd6d0c6230464a940c429291a4012

```

### Evaluating a pretrained model

Evaluating a pretrained model for a given dataset requires:
1. The pretrained experts for the target dataset, which should be located in `<root>/data/<dataset-name>/symlinked-feats`.
2. A `config.json` file.
3. A `trained_model.pth` file.

Evaluation is then performed with the following command:
```
python3 test.py --config <path-to-config.json> --resume <path-to-trained_model.pth> --device <gpu-id> --eval_from_training_config
```
where `<gpu-id>` is the index of the GPU to evaluate on.

### Training a new model

Training a new video-text embedding requires:
1. The pretrained experts for the dataset used for training, which should be located in `<root>/data/<dataset-name>/symlinked-feats`.
2. A `config.json` file.  You can define your own, or use one of the provided configs in the [configs](configs) directory.

Training is then performed with the following command:
```
python3 train.py --config <path-to-config.json> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to train on.  

### References

If you find this code useful or use the extracted features, please consider citing:

```
@inproceedings{Tran-et-al-ACCV22,
  author    = {Tran, Vinh and Balasubramanian, Niranjan and and Hoai, Minh},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  title     = {From Within to Between: Knowledge Distillation for Cross Modality Retrieval},
  month     = {December},
  date      = {2022},
}
```

### Acknowledgements

This codebase was built on top of *[TeachText](https://www.robots.ox.ac.uk/~vgg/research/teachtext/)* and *[Collaborative Experts](https://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/)*.
Many thanks to the authors.
