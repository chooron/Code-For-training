import pandas as pd


def load_data(name):
    data = pd.read_csv("data.csv")
    data.dropna(axis=0)
    pop = data[name].to_numpy()
    year = data['年份'].to_numpy()
    return year, pop


