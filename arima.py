#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import multiprocessing
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import arima
import pmdarima.model_selection
from sklearn.metrics import r2_score


def adjust_date(s):
    t = s.split("/")
    return f"20{t[2]}-{int(t[0]):02d}-{int(t[1]):02d}"


def adjust_name(s):
    return re.sub(r"[*,() ']", "_", s)

def draw(model, df, country):
    data = df[country].diff().dropna()
    model.fit(data)

    # 模型验证
    train, test = pmdarima.model_selection.train_test_split(data, train_size=0.8)
    pred_test = model.predict_in_sample(start=train.shape[0], dynamic=False)
    validating = pd.Series(pred_test, index=test.index)
    r2 = r2_score(test, pred_test)
    print(r2)

    # 开始预测
    pred, pred_ci = model.predict(n_periods=14, return_conf_int=True)
    idx = pd.date_range(data.index.max() + pd.Timedelta("1D"), periods=14, freq="D")
    forecasting = pd.Series(pred, index=idx)

    # 绘图呈现
    plt.figure(figsize=(24, 6))

    plt.plot(data.index, data, label="Actual Value", color="blue")
    plt.plot(validating.index, validating, label="Check Value", color="orange")
    plt.plot(forecasting.index, forecasting, label="Predict Value", color="red")
    # plt.fill_between(forecasting.index, pred_ci[:, 0], pred_ci[:, 1], color="black", alpha=.25)

    plt.legend()
    plt.ticklabel_format(style="plain", axis="y")
    # plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    # if isDaily:
    plt.title(
        f"Daily Confirmed Cases Forecasting - {country}\nARIMA {model.model_.order}x{model.model_.seasonal_order} (R2 = {r2:.6f})"
    )
    plt.savefig(
        os.path.join("figures", f"covid-{adjust_name(country)}-daily.svg"),
        bbox_inches="tight",
    )
    plt.close()

    #else:
    #    plt.title(
    #        f"Accumulative Confirmed Cases Forecasting - {country}\nARIMA {model.model_.order}x{model.model_.seasonal_order} (R2 = {r2:.6f})"
    #    )
    #    plt.savefig(
    #        os.path.join("figures", f"covid-{adjust_name(country)}.svg"),
    #        bbox_inches="tight",
    #    )
    #    plt.close()


if __name__ == "__main__":
    __spec__ = None
    # 准备数据
    df = (
        pd.read_csv(
            "time_series_covid19_confirmed_global.csv"
        )
            .drop(columns=["Lat", "Long"])
            .groupby("Country/Region")
            .sum()
            .transpose()
    )
    df.index = pd.DatetimeIndex(df.index.map(adjust_date))
    print("data prepared!")

    countries = df.columns.to_list()

    model = arima.AutoARIMA(
        start_p=0,
        max_p=4,
        d=None,
        start_q=0,
        max_q=1,
        start_P=0,
        max_P=1,
        D=None,
        start_Q=0,
        max_Q=1,
        m=7,
        seasonal=True,
        test="kpss",
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    # plt无法线程安全，使用异步进程
    def process_result(return_value):
        print(return_value)

    pool = multiprocessing.Pool()
    for i in range(len(countries)):
        pool.apply_async(draw, args=(model, df, countries[i]), callback=process_result)
    pool.close()
    pool.join()


    # 编制索引
    with codecs.open("ARIMA_County.md", "w", "utf-8") as f:
        f.write("# COVID-19 Forecasting\n\n")
        f.write(
            "[![Country application](https://github.com/Neteraxe/covid_seird/actions/workflows/country-app.yml/badge.svg)](https://github.com/Neteraxe/covid_seird/actions/workflows/country-app.yml)\n"
        )
        f.write(
            "[![Data Source](https://img.shields.io/badge/Data%20Source-https://github.com/CSSEGISandData/COVID--19-brightgreen)](https://github.com/CSSEGISandData/COVID-19)\n"
        )
        for country in countries:
            f.write(f"## {country}\n\n")
            f.write(f"![img](figures/covid-{adjust_name(country)}-daily.svg)\n\n")
