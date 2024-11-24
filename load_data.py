import zipfile


def load_data():
    # kaggle competitions download -c house-prices-advanced-regression-techniques
    with zipfile.ZipFile('house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')