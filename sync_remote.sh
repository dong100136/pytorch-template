rsync -au --progress --delete \
-e 'ssh -p 17431' \
--exclude-from=./exclude.list \
/Users/stoneye/github/pytorch-template \
root@10.106.128.107:/root/project/paper