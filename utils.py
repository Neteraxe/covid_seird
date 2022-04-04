import pandas as pd

def getAllJHU():
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for filename in ["time_series_covid19_deaths_global.csv", "time_series_covid19_recovered_global.csv", "time_series_covid19_confirmed_global.csv"]:
            print(executor.submit(dataTransform, filename).result())

def dataTransform(filename):
    df = (
        pd.read_csv(
            filename
        )
        .drop(columns=["Lat", "Long"])
        .groupby("Country/Region")
        .sum()
        .transpose()
    )
    df.index = pd.DatetimeIndex(df.index.map(adjust_date))
    df.to_csv(filename)

def adjust_date(s):
    t = s.split("/")
    return f"20{t[2]}-{int(t[0]):02d}-{int(t[1]):02d}"