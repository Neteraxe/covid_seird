import pandas as pd


def dataTransformAsyn():
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for filename in ["time_series_covid19_deaths_global.csv", "time_series_covid19_recovered_global.csv",
                         "time_series_covid19_confirmed_global.csv"]:
            executor.submit(dataTransformCountry, filename).result()
            executor.submit(dataTransformProvince, filename).result()


def dataTransformCountry(filename):
    df = (
        pd.read_csv(
            filename
        )
            .drop(columns=["Lat", "Long"])
            .groupby("Country/Region")
            .sum()
            .transpose()
    )
    df.index = pd.to_datetime(df.index)
    df.to_csv("country_" + filename)
    print(filename + " countries transform success!")


def dataTransformProvince(filename):
    df = pd.read_csv(
        filename,
        index_col="Province/State",
    ).drop(columns=["Lat", "Long"])
    df = (
        df[(df["Country/Region"] == "China") | (df["Country/Region"] == "Taiwan*")]
            .transpose()
            .drop("Country/Region")
            .rename(columns=str)
            .rename(columns={"nan": "Taiwan"})
            .drop(columns=["Unknown"])
            .sort_index(axis=1)
    )
    df.index = pd.to_datetime(df.index)
    df.to_csv("province_" + filename)
    print(filename + " provinces transfrom success!")


if __name__ == "__main__":
    dataTransformAsyn()
