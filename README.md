# Pneumonia EpiMed project
### Setup
First pip install all required packages
```
pip install -r requirements.txt
```
Setup a kaggle account and download API credentials by following the guide found [here](https://github.com/Kaggle/kaggle-api#api-credentials).

Change the path in setup_dataset.py to fit your local machine.
```
#####################################################
# Change to path to fit local filestructure
kaggle_info_path = '<YOUR LOCAL HOME DIR GOES HERE>.kaggle/kaggle.json'
#####################################################
``` 
Then run setup_dataset.py to download the dataset.
```
python setup_dataset.py
``` 