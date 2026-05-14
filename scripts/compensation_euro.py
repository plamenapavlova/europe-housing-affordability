import pandas as pd

df = pd.read_csv("../data/processed/compensation_employee.csv")
df['year'] = df['time'].astype(str).str[:4].astype(int)
print(df["Country"].unique())

df_rates = pd.read_excel("../data/raw/conversion_rates.xlsx")
df_rates.to_csv("../data/processed/conversion_rates.csv", index=False)

df_rates = df_rates.melt(id_vars="Country", var_name = 'year', value_name = 'exchange_rate')
df_rates['year'] = df_rates['year'].astype(int)

df = df.merge(
    df_rates[['Country', 'year', 'exchange_rate']],
    on=['Country', 'year'],
    how='left'
)
df["exchange_rate"] = df["exchange_rate"].fillna(1)
df['compensation'] = df['compensation']/df['exchange_rate']
df['compensation'] = df['compensation'].round(2)

#compensation per employee
df['com_per_employee'] = round(df['compensation'] / df['employment'], 1)

df.to_csv("../data/processed/compensation_employee_eur.csv", index=False)

