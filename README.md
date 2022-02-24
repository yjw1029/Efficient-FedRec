# Efficient-FedRec
Python implementation for our paper "Efficient-FedRec: Efficient Federated Learning Frameworkfor Privacy-Preserving News Recommendation" in EMNLP 2021.

## Introduction
Directly applying federated learning on news recommendation models will lead to high computation and communication cost on user side.
In this work, we propose Efficient-FedRec, in which we dicompose the news recommendation model into a large news model maintained on server and a light-weight user model computed on the user side.
Experiments on two public dataset show the effectiveness of our method.


## Environment
Requirments
```
numpy
torch==1.9.1
transformers==4.12.5
tqdm
sklearn
wandb
pycryptodome
cryptography
```

## Getting Started
* Install opacus
```bash
  cd opacus
  pip install -e .
```
* Download datasets 
```bash
cd raw
chmod +x download.sh
./download.sh mind .
./download.sh adressa .
```
* Preprocess datasets 
```bash
cd preprocess
# modify adressa to mind format
python adressa_raw.py

# preprocess mind dataset
python news_process.py --data mind
python user_process.py --data mind

# preprocess adressa dataset
python news_process.py --data adressa
python user_process.py --data adressa
```

* Run experiments
```bash
# You may need to configure your wandb account first
cd src
python main.py  --mode train --job_name Efficient-FedRec-MPC --run_name efficient-fedRec-mpc --data mind


# train on adressa
python main.py --mode train --job_name Efficient-FedRec-MPC --run_name efficient-fedRec-mpc --data adressa --max_train_steps 500 --validation_step 10 --bert_type NbAiLab/nb-bert-base
# test on adressa
python main.py --mode train --job_name Efficient-FedRec-MPC --run_name efficient-fedRec-mpc --data adressa --mode test --bert_type NbAiLab/nb-bert-base
```


## Results

### MIND 
Wandb result on MIND dataset
![](./.figure/mind-result.png)
Zip the prediction.txt file and upload to MIND competition. Test result is
![](./.figure/mind-leaderboard.png) 


## Adressa
Wandb result on Adressa dataset.
![](./.figure/adressa-result.png)
Test result is
```
test auc: 0.7980, mrr: 0.4637, ndcg5: 0.4852, ndcg10: 0.5497
```

## Citing
If you want to cite Efficient-Fedrec in your papers (much appreciated!), you can cite it as follows:
```
@inproceedings{yi-etal-2021-efficient,
    title = "Efficient-{F}ed{R}ec: Efficient Federated Learning Framework for Privacy-Preserving News Recommendation",
    author = "Yi, Jingwei  and
      Wu, Fangzhao  and
      Wu, Chuhan  and
      Liu, Ruixuan  and
      Sun, Guangzhong  and
      Xie, Xing",
    booktitle = "EMNLP",
    year = "2021",
    pages = "2814--2824"
}
```
