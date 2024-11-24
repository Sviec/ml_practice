import pandas as pd


def send_to_kaggle(model, df_selected, df):
    predictions = model.predict(df_selected).data.ravel()
    submission = pd.DataFrame({
        'Id': df['Id'],
        'SalePrice': predictions
    })

    submission.to_csv('submission.csv', index=False)