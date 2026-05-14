import pandas as pd
import matplotlib.pyplot as plt
from functions import rebase, long_format, assign_cluster, plot_by_group
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans

"""
The following code loads house price indexes data, rebases 
and converts to long format. Then, clusters countries based on
similar HPI dynamics.
"""

#nominal dynamics
df = pd.read_excel('../data/raw/hpi.xlsx')
df["Country"] = df["Country"].replace({"European Union ": "EU", "Euro area ": "EA"})
df.to_csv('../data/processed/hpi.csv', index=False)

df_long = long_format(df, 'hpi')
df_long = rebase(df_long, "hpi", "hpi_rebased")
df_long.to_csv('../data/processed/hpi_rebased.csv', index=False)
#print(df_long.head())
countries = ["Bulgaria", "EU", "EA"]
df_plot = df_long[df_long['Country'].isin(countries)]
df_plot = df_long

for country, g in df_plot.groupby("Country"):
    plt.plot(g["time"].astype(str), g["hpi_rebased"], label = country)

#plotting
#print(df_plot["Country"].unique())
years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
positions = [f"{y}Q1" for y in years]
plt.xticks(positions, years)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Index")
plt.title("Change in price index")
plt.show()

#only countries data without EU and Euro area
df_countries = df[~df["Country"].isin(["EU", "EA", "United Kingdom"])].copy()
df_countries = df_countries.reset_index(drop=True)
df_countries = long_format(df_countries, 'hpi')
df_countries = rebase(df_countries, 'hpi', 'hpi_rebased')
#print(df_countries.head())

df_wide = df_countries.pivot(index="Country", columns="time", values="hpi_rebased")

dataset = df_wide.to_numpy()
dataset = dataset[:, :, None]

model = TimeSeriesKMeans(n_clusters = 4, metric='dtw', random_state=0)
labels_scaled = model.fit_predict(dataset)

cluster_dict = assign_cluster(df_wide, labels_scaled)
#print_clusters(cluster_dict)
plot_by_group(df_countries, cluster_dict, 'hpi_rebased', 'cluster', 'HPI (rebased)')


