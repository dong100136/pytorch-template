# python3 train.py -c config/unbalance/GHM/camelyon16_resnet50_GHM_20.yaml -d 0
# python3 train.py -c config/unbalance/GHM/camelyon16_resnet50_GHM_10.yaml -d 0
#
python3 train.py -c config/unbalance/GHM/camelyon16_densenet161_GHM_20.yaml -d 0
python3 train.py -c config/unbalance/GHM/camelyon16_densenet161_GHM_10.yaml -d 0

python3 train.py -c config/unbalance/weights/camelyon16_densenet161_weights_1_2.yaml -d 0
python3 train.py -c config/unbalance/weights/camelyon16_densenet161_weights_1_3.yaml -d 0
python3 train.py -c config/unbalance/weights/camelyon16_densenet161_weights_2_1.yaml -d 0
python3 train.py -c config/unbalance/weights/camelyon16_densenet161_weights_3_1.yaml -d 0
