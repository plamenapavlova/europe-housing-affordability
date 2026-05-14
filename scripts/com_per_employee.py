import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from project.scripts.functions import long_format, rebase, assign_cluster, plot_by_group, plot_by_country, \
    long_format_hicp
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

"""
Calculates a housing affordability index by using compensation per
employee and then clusters based on affordability patterns and 
visualizes the time-series plots and geographic map.
"""

cmap = plt.cm.Set2
colors = {
    0: cmap(0),
    1: cmap(1),
    2: cmap(2),
    3: cmap(3)
}

#df_com = pd.read_excel('../data/raw/compensation.xlsx')
#df_com = long_format(df_com, 'compensation')

#df_emp = pd.read_excel('../data/raw/employment.xlsx')
#df_emp = long_format(df_emp, 'employment')

#merge
#df = df_com.merge(df_emp, on = ['Country', 'time'])
#df = df[df['Country'] != 'Greece']

df = pd.read_csv('../data/processed/compensation_employee_eur.csv')
df['time'] = pd.PeriodIndex(df['time'], freq='Q')
#real compensation per employee
df_hicp = pd.read_csv('../data/processed/hicp.csv')

df_hicp = long_format_hicp(df_hicp)
df_hicp = (df_hicp.groupby(['Country', pd.Grouper(key = 'time', freq = 'QE')])['hicp'].mean().reset_index())
df_hicp['time'] = df_hicp['time'].dt.to_period('Q')
df_hicp['hicp'] = df_hicp['hicp'].round(2)

#merging
if 'hicp' in df.columns:
    df = df.drop(columns=['hicp'])
df = df.merge(df_hicp[['Country', 'time', 'hicp']], on=['Country', 'time'], how='left')
df['com_per_employee'] = df['com_per_employee']/df['hicp']*100
df = rebase(df, "com_per_employee", "comp_emp_rebased")
df.to_csv('../data/processed/compensation_employee_real_rebased.csv', index = False)


#wide format
df_wide = df.pivot(index='Country', columns='time', values='comp_emp_rebased')

#Load hpi file
df_hpi = pd.read_csv('../data/processed/real_hpi.csv')
df_hpi['time'] = pd.PeriodIndex(df_hpi['time'], freq='Q')


df_affordability = df_hpi.merge(df[['Country', 'time', 'comp_emp_rebased']], on = ['Country', 'time'])
df_affordability['affordability'] = (df_affordability['comp_emp_rebased']/df_affordability['rhpi_rebased']*100).round(2)
#print(df_affordability.head())
df_affordability.to_csv('../data/processed/affordability.csv', index = False)

df_wide = df_affordability.pivot(index='Country', columns='time', values='affordability')
df_wide = df_wide.sort_index(axis=1)
df_wide = df_wide.dropna()
dataset = df_wide.to_numpy()
dataset = dataset[:, :, None]

scaler = TimeSeriesScalerMeanVariance()
dataset_scaled = scaler.fit_transform(dataset)

#similar behaviour
model = TimeSeriesKMeans(n_clusters = 4, metric = 'dtw')
labels = model.fit_predict(dataset_scaled)
cluster_dict = assign_cluster(df_wide, labels)
#plot_by_group(df_affordability, cluster_dict, "affordability", "cluster", "affordability index by behaviour")

#similar values
model = TimeSeriesKMeans(n_clusters = 4, metric = 'euclidean')
labels_euc = model.fit_predict(dataset)
cluster_dict_euc = assign_cluster(df_wide, labels_euc)
#plot_by_group(df_affordability, cluster_dict_euc, "affordability", "cluster", "affordability index by level")


country_to_cluster_euc = {country: int(cluster) for cluster, countries in cluster_dict_euc.items() for country in countries}
country_to_cluster_dtw = {country: int(cluster) for cluster, countries in cluster_dict.items() for country in countries}

world = gpd.read_file('../data/raw/ne_10m_admin_0_countries.zip')
europe = world[world["CONTINENT"] == "Europe"].copy()

# filter Europe
europe = world[world["CONTINENT"] == "Europe"].copy()
europe['cluster_euc'] = europe['NAME'].map(country_to_cluster_euc)
europe['cluster_euc'] = europe["cluster_euc"].astype("Int64")

europe['cluster_dtw'] = europe['NAME'].map(country_to_cluster_dtw)
europe['cluster_dtw'] = europe["cluster_dtw"].astype("Int64")


bbox = gpd.GeoDataFrame(geometry=[box(-25, 30, 45, 75)],crs=world.crs)
europe = gpd.clip(europe, bbox)
europe["color_euc"] = europe["cluster_euc"].map(colors).fillna("lightgrey")
europe["color_dtw"] = europe["cluster_dtw"].map(colors).fillna("lightgrey")

fig_map, axes = plt.subplots(1, 2, figsize=(10, 8))
europe.plot(color = europe['color_euc'], ax=axes[0], edgecolor="black", linewidth=0.5)
axes[0].set_xlim(-25, 45)
axes[0].set_ylim(30, 75)
axes[0].set_title("Absolute Level Clusters (Euclidean)", fontsize=16, pad=10)
axes[0].axis("off")
plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)


#dtw map
europe.plot(  color=europe["color_dtw"], ax=axes[1], edgecolor="black", linewidth=0.5)
axes[1].set_xlim(-25, 45)
axes[1].set_ylim(30, 75)
axes[1].set_title("Pattern Clusters (DTW)", fontsize=16, pad=10)
axes[1].axis("off")

fig_map.suptitle("Housing Affordability in Europe", fontsize =20)
plt.tight_layout()
plt.show()



df_affordability["cluster_euc"] = df_affordability["Country"].map(country_to_cluster_euc)
df_affordability["cluster_dtw"] = df_affordability["Country"].map(country_to_cluster_dtw)
#mean of clusters over time
mean_euc = (df_affordability.groupby(["cluster_euc", "time"])['affordability'].mean().reset_index())
mean_dtw = (df_affordability.groupby(['cluster_dtw', 'time'])['affordability'].mean().reset_index())


#plotting clusters over time
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle("Mean affordability through time",fontsize =20 )
for cluster, g in mean_euc.groupby("cluster_euc"):
    cluster = int(cluster)
    axes[0].plot(g["time"].astype(str), g["affordability"], label=f"Cluster {cluster}", color=colors[cluster])


years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
positions = [f"{y}Q1" for y in years]
axes[0].set_xticks(positions, years)
axes[0].legend()
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Mean Affordability Index")
axes[0].set_title("By absolute level (Euclidean Cluster)")

for cluster, g in mean_dtw.groupby("cluster_dtw"):
    cluster = int(cluster)
    axes[1].plot(g["time"].astype(str), g["affordability"], label=f"Cluster {cluster}", color=colors[cluster])

axes[1].set_xticks(positions, years)
axes[1].legend()
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Mean Affordability Index")
axes[1].set_title("By patterns (DTW Cluster)")
plt.tight_layout()
plt.show()

# get countries in cluster 1
# Find Bulgaria's cluster and plot all countries in it
bulgaria_cluster = country_to_cluster_euc['Bulgaria']
bulgaria_cluster_countries = [country for country, cluster in country_to_cluster_euc.items() if cluster == bulgaria_cluster]
df_bulgaria_cluster = df_affordability[df_affordability['Country'].isin(bulgaria_cluster_countries)]

# Bulgaria's affordability in its affordability cluster
plot_by_country(df_bulgaria_cluster, "affordability", "Affordability Index",
                f"Cluster {bulgaria_cluster} - Affordability by Country")

# Bulgaria's affordability in its real HPI cluster
rhpi_clusters = pd.read_csv('../data/processed/rhpi_clusters.csv')
country_to_cluster_rhpi = dict(zip(rhpi_clusters['Country'], rhpi_clusters['cluster_rhpi']))
bulgaria_rhpi_cluster = country_to_cluster_rhpi['Bulgaria']
bulgaria_rhpi_countries = [c for c, cl in country_to_cluster_rhpi.items() if cl == bulgaria_rhpi_cluster]
df_bulgaria_rhpi = df_affordability[df_affordability['Country'].isin(bulgaria_rhpi_countries)]
plot_by_country(df_bulgaria_rhpi, "affordability", "Affordability Index",
                f"Real HPI Cluster {bulgaria_rhpi_cluster} - Affordability by Country")

# Plot real compensation per employee without rebasing
plot_by_country(df[['Country', 'time', 'com_per_employee']],
                "com_per_employee",
                "Real Compensation per Employee (thousand EUR, quarterly)",
                "Real Compensation per Employee by Country")


bg = df_affordability[df_affordability["Country"] == "Bulgaria"]
print(bg[["time", "comp_emp_rebased", "rhpi_rebased", "affordability"]].tail())

bg = df_affordability[df_affordability["Country"] == "Bulgaria"]

plt.figure(figsize=(10, 6))
plt.plot(bg["time"].astype(str), bg["comp_emp_rebased"], label="Real compensation per employee")
plt.plot(bg["time"].astype(str), bg["rhpi_rebased"], label="Real house price index")

years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
positions = [f"{y}Q1" for y in years]
plt.xticks(positions, years)

plt.axhline(100, linestyle="--", linewidth=1)
plt.xlabel("Year")
plt.ylabel("Index, 2010 = 100")
plt.title("Bulgaria: Real Compensation vs Real House Prices")
plt.legend()
plt.show()






