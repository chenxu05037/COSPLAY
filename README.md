# COSPLAY

This is a messy version code (under testing) for the
paper [COSPLAY: Concept Set Guided Personalized Dialogue Generation Across Both Party Personas](https://dl.acm.org/doi/abs/10.1145/3477495.3531957) (SIGIR 2022).

## Environment

We provide the conda
env ([cosplay.zip](https://drive.google.com/file/d/1RPcT7QCmUxZ9J7sVS13cfGZUIT6GbWI6/view?usp=share_link)) to setup
environment.
With conda installed, create cosplay conda environment by:

```
unzip cosplay.zip
conda create --name cos --clone cosplay
conda activate cos
```

## Data
The dataset and the preprocessed data for concept set framework can be found here ([data.zip]()).
```
unzip data.zip
```

## Train COSPLAY in Supervised Learning

```
python train_cosplay_in_supervised.py
```

## Train COSPLAY in Reinforced Learning

```
python train_cosplay_in_reinforced.py
```

## Evaluate COSPLAY
In addition to the above training from scratch, our pre-trained cosplay can be found [here](). 
Then test model by the evaluation codes.

## Citation

If you use this codebase in your work, please consider citing our paper:

```
@inproceedings{xu2022cosplay,
  title={COSPLAY: Concept Set Guided Personalized Dialogue Generation Across Both Party Personas},
  author={Xu, Chen and Li, Piji and Wang, Wei and Yang, Haoran and Wang, Siyun and Xiao, Chuangbai},
  journal={The 45th International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  pages={201--211},
  year={2022}
}
```