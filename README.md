# DeepOTG

The structure of this project is:


## Installation Guide
1.Create a conda environment and activate it:
```
conda create -n DeepOTG python==3.12.3
conda activate DeepOTG
```

2.install the environment:


## Usage
Owe to the upload limit, unzip the tar files in `db` folder first.
```
tar -xvf train.tar
tar -xvf balanced.tar
tar -xvf imbalanced.tar
```

1.To validate our results:
```
python validate.py
```

2.To predict your samples, generate PSSM and ESM features first:
Run `python generate_esm_features.py` to generate ESM features, and the PSSM file generator is provided in folder db as `get_pssm`.
As files name assumed, the structure should be: 


```
python predict.py
```



3.To train your models:
```
python train.py
``` 
And the models will be saved in folder 'models' and the logs will be generated in folder 'log'.

