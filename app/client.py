import requests
import pandas as pd


if __name__ == '__main__':
    X = pd.read_csv('app/data/data_new.csv')
    X.drop('target', axis=1, inplace=True)

    feature = list(X.loc[0].values)

    r = requests.post('http://0.0.0.0:8000/predict', json=feature)
    print(f'Status: {r.status_code}')

    if r.status_code == 200:
        print(f"Prediction: {r.json()}")   
    else:
        print(r.text) 