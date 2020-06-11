#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
*__pycache__*
.git/*
*.pth*
.vscode*
EOM

if [ "$1" == "hust" ]; then

    echo "Push data to hust (2x2080Ti)"
    rsync -vr -P --exclude-from $TEMP_FILE "/mnt/DATA/learning_stuffs/uni/20192/artificial_intelligence/project/alpha-zero/source/" hust:/home/hieu123/alpha_zero_mp/
elif [ "$1" == "workstation" ]; then

    echo "Push data to workstaion (1x2080Ti)"
    rsync -vr -P --exclude-from $TEMP_FILE "/mnt/DATA/learning_stuffs/uni/20192/artificial_intelligence/project/alpha-zero/source/" workstation:/home/ad/Documents/source/

else
    echo "Unknown server"
    exit
fi

# push code to server
# rsync -vr -P --exclude-from $TEMP_FILE "/mnt/DATA/learning_stuffs/uni/20192/artificial_intelligence/project/alpha-zero/source/" $1:$2

# rsync -vr -P --exclude-from $TEMP_FILE "/mnt/DATA/learning_stuffs/uni/20192/artificial_intelligence/project/alpha-zero/source/" hust:/home/hieu123/alpha_zero_mp/
# rsync -vr -P --exclude-from $TEMP_FILE "$PWD" $1:$REMOTE_HOME
# rsync -vr -P --exclude-from $TEMP_FILE /mnt/DATA/mammo/source/ workstation:/home/ad/mammo_detection/source/
# pull model weights and log files from server
# rsync -vr -P -e "ssh -p$PORT $JUMP" $USER@$IP:$REMOTE_HOME/nhan/$PROJECT_NAME/logs/*.txt ./logs/
# rsync -vr -P -e "ssh -p$PORT $JUMP" $USER@$IP:$REMOTE_HOME/nhan/$PROJECT_NAME/weights/best* ./weights/
# remove temp. file
rm $TEMP_FILE