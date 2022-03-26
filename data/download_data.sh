#!/bin/bash

dataset='imagewoof2-160'
data_dir=/content/drive/MyDrive/classification_dogs/data
wget https://s3.amazonaws.com/fast-ai-imageclas/$dataset.tgz
mkdir -p $data_dir
tar -xzf $dataset.tgz -C $data_dir
rm -rf $dataset.tgz
wget https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt -P $data_dir/$dataset
