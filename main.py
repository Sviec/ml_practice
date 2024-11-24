import zipfile
import pandas as pd
from modeling import modeling
from feature_engineering import feature_engineering
from preprocess import preprocess
from send_to_kaggle import send_to_kaggle


def main():
    # Загрузка данных
    load_data()
    # Чтение данных
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    # Препроцессинг (удаление фичей с более чем 30% пропущенных значений,
    # заполенение пустых значений модой у категориальных фичей и медианой у числовых
    train_data_selected = train_data.copy()
    train_data_selected = preprocess(train_data_selected)

    # Добавление новых фичей и кодирование категориальных
    train_data_selected = feature_engineering(train_data_selected)

    # Обучение модели на подготовленном датасете
    automl, oof_pred = modeling(train_data_selected)

    # Отправка результатов в файл submission
    test_data_selected = test_data.copy()
    test_data_selected = preprocess(test_data_selected)
    test_data_selected = feature_engineering(test_data_selected)

    send_to_kaggle(automl, test_data_selected, test_data)


def __init__():
    main()