# DeepOTG

The structure of this project is:
```text
DeepOTG/
├── data/
├── db/
├── log/ 
├── saved_models/   
├── generate_esm.py  
├── load_data.py 
├── model.py   
├── predict.py   
├── train.py    
└── validate.py 
```

## Installation Guide
1.Create a conda environment and activate it:
```
conda create -n DeepOTG python==3.12.3
conda activate DeepOTG
```

2.install the environment:
```
pip install numpy==2.1.3 pandas==2.3.1 scikit-learn==1.6.1 torch==2.5.1 transformers
```

## Usage
Due to the upload limit, unzip the tar files in the `db` folder first.
```
tar -xvf train.tar
tar -xvf balanced.tar
tar -xvf imbalanced.tar
```

### 1.To validate our results:
```
python validate.py
```

### 2.To predict your samples:
Generate PSSM and ESM features first:

Run `python generate_esm_features.py` to generate ESM features, and the PSSM file generator is provided in folder db as `get_pssm`.

As the code assumes, the structure should be: 

- **Sequence File** (`data/train.txt` or similar): Must be in FASTA format.
  - The header line must contain **"pos"** (case insensitive) to be recognized as a positive sample.
  - Any header without "pos" will be treated as a negative sample.
  ```text
  >SequenceID|Label
  STGQA...
  ```
- **PSSM Files**: Must be located in the specified directory (e.g., `db/train/pssm/`), or you can change the file path in the `predict.py`.
  - Filename must match the sequence ID: `{SequenceID}.pssm`
- **ESM Features**: Generated via the script below.

  To generate ESM features:
  ```bash
  python generate_esm_features.py
  ```

To run predictions:
```bash
python predict.py
```


### 3.To train your models:
Ensure your data is prepared in `data/train.txt`, ESM features are in `data/train_esm.txt`, and PSSM files are in `db/train/pssm`.

```
python train.py
``` 
The models will be saved in folder `saved_models` (specifically as `fold_*.pth`) and the logs will be generated in folder `log`.
