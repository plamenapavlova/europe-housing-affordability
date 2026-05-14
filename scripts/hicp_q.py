import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from functions import long_format, long_format_hicp, rebase, plot_by_group, assign_cluster, print_clusters, \
    plot_by_country

"""
The code loads HICP data, aligns with the HPI data, converts to 
quarters, calculates and rebases inflation-adjusted real house 
price indices. After that it clusters countries by similar 
housing price dynamics. 
"""

df = pd.read_excel('../data/raw/hicp.xlsx', index_col = 0)
#clean data
df = df.drop(columns = [col for col in df.columns if 'Unnamed' in col])
df = df.reset_index()

#remove unnecessary countries
df_hpi = pd.read_csv('../data/processed/hpi.csv')
countries = df_hpi['Country'].unique()
df = df[df['Country'].isin(countries)]
df.to_csv('../data/processed/hicp.csv', index=False)
df_hpi = long_format(df_hpi, 'hpi')

#convert into quarters
df = long_format_hicp(df)
df = (df.groupby(['Country', pd.Grouper(key = 'time', freq = 'QE')])['hicp'].mean().reset_index())
df['time'] = df['time'].dt.to_period('Q')
df['hicp'] = df['hicp'].round(2)

#get real hpi - Inflation-adjusted house prices
real_hpi = df.merge(df_hpi, on = ["Country", "time"])
real_hpi["rhpi"] = real_hpi['hpi']/real_hpi['hicp']
real_hpi["rhpi"] = real_hpi["rhpi"].round(2)
real_hpi = real_hpi.drop(columns = ['hpi', 'hicp'])
real_hpi.to_csv('../data/processed/real_hpi.csv')


#rebase
real_hpi =rebase(real_hpi, 'rhpi', 'rhpi_rebased')
real_hpi.to_csv('../data/processed/real_hpi.csv')
#plot
#plot_by_country(real_hpi, "rhpi_rebased", "Real HPI", "Real house price index")

#splitting by regions
regions = {
    "Central & Eastern Europe":{'Czechia', 'Hungary', 'Poland', 'Bulgaria', 'Romania',
                                'Croatia', 'Slovakia', 'Slovenia'},
    "Southern Europe":{'Cyprus','Italy', 'Malta', 'Portugal', 'Spain'},
    "Northern Europe":{'Denmark', 'Estonia', 'Finland', 'Iceland', 'Latvia', 'Lithuania', 'Norway', 'Sweden'},
    "Western Europe":{'Austria', 'Belgium', 'France', 'Germany', 'Luxembourg', 'Netherlands', 'Ireland' }
}

#plot_by_group(real_hpi, regions, "rhpi_rebased", "region", 'Real HPI')

#clustering
df_wide = real_hpi.pivot(index="Country", columns="time", values="rhpi_rebased")
df_wide = df_wide.dropna()

dataset = df_wide.to_numpy()
dataset = dataset[:, :, None]

#cluster only on similar shape - no matter the timing “who behaves similarly?” housing cycles
scaler = TimeSeriesScalerMeanVariance()
dataset_scaled = scaler.fit_transform(dataset)
model = TimeSeriesKMeans(n_clusters = 4, metric='dtw', random_state=0)
labels_scaled = model.fit_predict(dataset_scaled)

cluster_dict = assign_cluster(df_wide, labels_scaled)
#print_clusters(cluster_dict)
#plot_by_group(real_hpi, cluster_dict, 'rhpi_rebased', "cluster", 'Real HPI')

#validation
score_dtw = silhouette_score(dataset_scaled, labels_scaled, metric='dtw')
print("Validation using silhouette score of dtw", f"{score_dtw:.2f}")

#cluster based on similar values “who grows similarly?” housing growth
model = TimeSeriesKMeans(n_clusters = 4, metric='euclidean', random_state=0)
labels = model.fit_predict(dataset)
print(f"No scaling and euclidean distance:")
cluster_map = assign_cluster(df_wide, labels)
print_clusters(cluster_map)
plot_by_group(real_hpi, cluster_map, "rhpi_rebased", "cluster", 'Real HPI')

#validation
score_euc = silhouette_score(dataset, labels, metric='euclidean')
print(f"Validation using silhouette score of euclidean", f"{score_euc:.2f}")













