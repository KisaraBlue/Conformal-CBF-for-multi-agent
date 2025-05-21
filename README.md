# Safe Decentralized Multi-Agent Control Using Black-Box Predictors, Conformal Decision Policies, and Control Barrier Functions

This the official repisotory for [Safe Decentralized Multi-Agent Control Using Black-Box Predictors, Conformal Decision Policies, and Control Barrier Functions](https://arxiv.org/pdf/2409.18862) by Sacha Huriot and Hussein Sibai.

## Setup

### Stanford Drone Dataset
Follow instructions on https://www.kaggle.com/datasets/aryashah2k/stanford-drone-dataset to download the dataset

You also need to download [`ynet_additional_files`](https://drive.google.com/file/d/1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa/view?usp=sharing). 

Then you need to edit the default arguments to [`load_SDD`](https://github.com/Jordylek/conformal-decision/blob/d3f3e97157d7f1ce0957cbba910a699be3f16f8b/sdd/utils/preprocessing.py#L14) to point to these filepaths.

You then need to create a cache for the predictions of the humans' next positions
```
    bash sdd/bash-cache-darts.py
```

Then you can create the trajectory for the robot and generate the video

```
    bash bash-traj-sacha.sh
```

The videos will be stored in `sdd/videos` and the results in `sdd/metrics`.

### Citation 

```
@misc{huriot2024safedecentralizedmultiagentcontrol,
      title={Safe Decentralized Multi-Agent Control using Black-Box Predictors, Conformal Decision Policies, and Control Barrier Functions}, 
      author={Sacha Huriot and Hussein Sibai},
      year={2024},
      eprint={2409.18862},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2409.18862}, 
}
```
