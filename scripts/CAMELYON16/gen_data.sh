python3 extra/CAMELYON16/gen_data.py \
~/dataset/CAMELYON16/testing \
/root/dataset/workspace/CAMELYON16_v2/valid \
--random true

# python3 extra/CAMELYON16/gen_data.py \
# ~/dataset/CAMELYON16/testing \
# /root/dataset/workspace/CAMELYON16_v2/valid_wsi \
# --random false

# python3 extra/CAMELYON16/gen_data.py \
# ~/dataset/CAMELYON16/training \
# /root/dataset/workspace/CAMELYON16_v2/train \
# --random true


# #----
python3 extra/CAMELYON16/gen_patch.py \
/root/dataset/workspace/CAMELYON16_v2/valid/normal_all_data.list \
/root/dataset/workspace/CAMELYON16_v2/valid/normal


python3 extra/CAMELYON16/gen_patch.py \
/root/dataset/workspace/CAMELYON16_v2/valid/tumor_all_data.list \
/root/dataset/workspace/CAMELYON16_v2/valid/tumor

# #----
# python3 extra/CAMELYON16/gen_patch.py \
# /root/dataset/workspace/CAMELYON16_v2/valid_wsi/normal_all_data.list \
# /root/dataset/workspace/CAMELYON16_v2/valid_wsi/normal

# python3 extra/CAMELYON16/gen_patch.py \
# /root/dataset/workspace/CAMELYON16_v2/valid_wsi/tumor_all_data.list \
# /root/dataset/workspace/CAMELYON16_v2/valid_wsi/tumor

# #----
# python3 extra/CAMELYON16/gen_patch.py \
# /root/dataset/workspace/CAMELYON16_v2/train/normal_all_data.list \
# /root/dataset/workspace/CAMELYON16_v2/train/normal

# python3 extra/CAMELYON16/gen_patch.py \
# /root/dataset/workspace/CAMELYON16_v2/train/tumor_all_data.list \
# /root/dataset/workspace/CAMELYON16_v2/train/tumor