BASE_MODEL=/root/trash/log/model/SaltNet-v9.1/model_best.pth
python3 train.py -c config/kaggle/tgs-salt-identitification/tgs_fold0.yaml -r $BASE_MODEL
python3 train.py -c config/kaggle/tgs-salt-identitification/tgs_fold1.yaml -r $BASE_MODEL
python3 train.py -c config/kaggle/tgs-salt-identitification/tgs_fold2.yaml -r $BASE_MODEL
python3 train.py -c config/kaggle/tgs-salt-identitification/tgs_fold3.yaml -r $BASE_MODEL
python3 train.py -c config/kaggle/tgs-salt-identitification/tgs_fold4.yaml -r $BASE_MODEL