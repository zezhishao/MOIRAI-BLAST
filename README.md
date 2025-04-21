# How to train MOIRAI with BLAST

**1. Place the training data in the `preprocess_data/raw_blast` directory.**

**2. Run the following command to convert the data:**
```bash
python preprocess_data/convert_data.py
```

**3. The converted data will be saved in the `example_dataset_1/blast` directory.**

**4. Train MOIRAI**
```bash
python cli/train.py # modify the config file to change differet version of MOIRAI
```
