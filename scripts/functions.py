import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

"""
This file provides helper functions for rebasing, converting
to long format, plotting and clustering time-series index data.
"""

def rebase(df: pd.DataFrame, value_col:str, new_col:str):
    """
    Rebases country's indexes so that the first observation equals 100.
    :param df: the dataframe containing indexes column to be rebased
    :param value_col: name of the column to be rebased
    :param new_col: name of the new column that will store the rebased values
    :return: the df with rebased column
    """
    df[new_col] = (df[value_col] / df.groupby("Country")[value_col].transform("first")) * 100
    df = df.round(2)
    return df

def long_format(df: pd.DataFrame, col_name):
    """
    Converts from wide format to long format
    :param df:wide-format dataframe with house price index values
    :return:long-format dataframe with columns - Country, time, hpi
    """
    df_long = df.melt(id_vars="Country", var_name="time", value_name=col_name)
    df_long["time"] = pd.PeriodIndex(df_long["time"], freq="Q")
    df_long[col_name] = pd.to_numeric(df_long[col_name], errors="coerce")
    df_long = df_long.sort_values(["Country", "time"])
    return df_long

def long_format_hicp(df):
    """
    Convets HICP data from wide format to long format
    :param df: wide-format dataframe with HICP values
    :return:ong-format dataframe with columns - Country, time, hicp
    """
    df_long = df.melt(id_vars = 'Country', var_name = 'time', value_name = 'hicp')
    df_long['time'] = pd.to_datetime(df_long["time"], format = '%Y-%m', errors='coerce')
    df_long['hicp'] = pd.to_numeric(df_long['hicp'], errors="coerce")
    return df_long

def plot_by_country(df: pd.DataFrame, value_col:str, y_label, title):
    """
    Plots time series by country
    :param df: dataframe containing Country, time and indexes values
    :param value_col: name of the column to be plotted
    :param y_label: label for the y-axis
    :param title: title of the plot
    :return:displays a plot with the countries
    """
    countries = df['Country'].unique()
    colors = plt.cm.tab20(range(len(countries)))
    for (country, g), color in zip(df.groupby('Country'),  colors):
        plt.plot(g["time"].astype(str), g[value_col], label=country, color=color)

    years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    positions = [f"{y}Q1" for y in years]
    plt.xticks(positions, years)
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def plot_by_group(df, groups, value_col, group_type, y_label):
    """
    Plots time series by region using plot_by_country
    :param df: dataframe containing Country, time and indexes values
    :param groups: dictionary mapping groups with the countries in it
    :param value_col: name of the column to be plotted
    :param group_type: cluster or region
    :return: displays one plot per region
    """
    for group in sorted(groups.keys()):
        countries = groups[group]
        df_region = df[df['Country'].isin(countries)]
        plt.figure()
        if group_type == "cluster":
            title = f"Cluster {group}"
        else:
            title = group
        plot_by_country(df_region, value_col, y_label, title)

def assign_cluster(df, labels):
    """
    Assigns cluster labels to each country
    :param df: datagrame containing countries
    :param labels: labels of each country
    :return: Clusters with assigned countries
    """
    cluster_dict = defaultdict(list)
    for country, labels in zip(df.index, labels):
        cluster_dict[labels].append(country)
    return dict(cluster_dict)


def print_clusters(cluster_dict):
    """
    Prints clusters with their values
    :param cluster_dict: dictionary with clusters and corresponding countries
    :return: None
    """
    for cluster in sorted(cluster_dict.keys()):
        print(f"Cluster {cluster}: {cluster_dict[cluster]}")