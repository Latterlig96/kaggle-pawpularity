# kaggle-pawpularity
Repo for ongoing pawpularity competition hosted on Kaggle

# How to run
After cloning repository just type `pip install -r requirements.txt`, then in config folder just adjust parameters to run either single model training, ensemble training or <strong>special</strong> resizer model training. After that just type `python -m pawpularity --mode <your_desired_train_mode>`

# CV
With 2 level ensembling (ViT+Swin->SVR->Ridge Regression) I was able to reach CV RMSE 17.52 that resulted in PublicLeaderBoard/PrivateLeaderBoard -> 18.05/17.24

# Standing
With current repository I was able to reach 493th position (top 15%).

# Further improvements
As the competition reached to its end, I do not intend to keep improving this repository.