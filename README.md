## Dog breed classification

### Dataset: 
  * [Imagewoof](https://github.com/fastai/imagenette#imagewoof)
> Imagewoof is a subset of 10 classes from Imagenet that aren't so easy to classify, since they're all dog breeds. The breeds are: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, Old English sheepdog. 

### Usage:
- [Colab](https://colab.research.google.com/drive/1tP1FnphFP3IviG3pqhoi5pBFvp5gPhy8?usp=sharing)


### Experimental results:
....

<details>
  <summary>Training</summary>

**Environment:**
```bash
https://github.com/adeshkin/classification_dogs.git 
cd classification_dogs
python3 -m venv ./venv
source venv/bin/activate
pip install -r scripts/requirements.txt
```

**Data preparation:**
```bash
cd data
bash download_data.sh
python3 prepare_data.py
```

**Training:**
```bash
cd scripts
python3 train.py /path/to/config
```
</details>