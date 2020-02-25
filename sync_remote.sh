rsync -au --progress --delete \
-e 'ssh -p 17459' \
--exclude-from=./exclude.list \
/Users/stoneye/github/pytorch-template \
root@10.106.128.105:/root/project/paper