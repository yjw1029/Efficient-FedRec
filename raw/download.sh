#!/bin/bash

DATASET=$1
DIR=$2

if [ $DATASET == "mind" ]
then
    mkdir $DIR/mind
    mkdir $DIR/mind/train
    mkdir $DIR/mind/valid
    wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip -P $DIR/mind/train
    unzip $DIR/mind/train/MINDsmall_train.zip -d $DIR/mind/train/
    rm $DIR/mind/train/MINDsmall_train.zip
    wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip -P $DIR/mind/valid
    unzip $DIR/mind/valid/MINDsmall_dev.zip -d $DIR/mind/valid/
    rm $DIR/mind/valid/MINDsmall_dev.zip
    wget https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip -P $DIR/mind/test
    unzip $DIR/mind/test/MINDlarge_test.zip -d $DIR/mind/test/
    rm $DIR/mind/test/MINDlarge_test.zip
elif [ $DATASET == "adressa" ]
then
    mkdir $DIR/adressa
    mkdir $DIR/adressa/raw
    wget --no-check-certificate http://reclab.idi.ntnu.no/dataset/one_week.tar.gz -P $DIR/adressa/raw
    tar zvfx $DIR/adressa/raw/one_week.tar.gz -C $DIR/adressa/raw
    rm $DIR/adressa/raw/one_week.tar.gz
fi
