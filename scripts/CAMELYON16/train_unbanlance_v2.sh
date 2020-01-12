# python3 train.py -c config/unbalance/GHM/camelyon16_resnet50_GHM_40.yaml -d 1
# python3 train.py -c config/unbalance/GHM/camelyon16_resnet50_GHM_30.yaml -d 1

python3 train.py -c config/unbalance/GHM/camelyon16_densenet161_GHM_40.yaml -d 1
python3 train.py -c config/unbalance/GHM/camelyon16_densenet161_GHM_30.yaml -d 1


# python3 train.py -c config/unbalance/weights/camelyon16_resnet50_weights_1_2.yaml -d 1
# python3 train.py -c config/unbalance/weights/camelyon16_resnet50_weights_1_3.yaml -d 1
# python3 train.py -c config/unbalance/weights/camelyon16_resnet50_weights_2_1.yaml -d 1
# python3 train.py -c config/unbalance/weights/camelyon16_resnet50_weights_3_1.yaml -d 1
