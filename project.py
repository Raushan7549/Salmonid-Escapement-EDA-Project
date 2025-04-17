import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\raush\\Downloads\\WDFW-Salmonid_Population_Indicators__SPI__Escapement_20250405.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print("\n Dataset Overview:")
print(df)

print("\n Head of the dataset:")
print(df.head())

print("\n Tail of the dataset:")
print(df.tail())

print("\n Summary Statistics:")
print(df.describe())

print("\n Information:")
print(df.info())

print("\n Column Names:")
print(df.columns)

print("\n Shape of Dataset:")
print(df.shape)

print("\n Null Values:")
print(df.isnull().sum())

#OBJECTIVE 1: Trend Analysis of Salmon Escapement by Species (Year-wise, with 3-Year Rolling Average);

df['abundance_quantity'] = pd.to_numeric(df['abundance_quantity'], errors='coerce')

df = df[df['abundance_quantity'].notnull() & (df['abundance_quantity'] > 0)]

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

grouped = df.groupby(['year', 'species'])['abundance_quantity'].sum().reset_index()

grouped['rolling_avg'] = grouped.groupby('species')['abundance_quantity'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean())

plt.figure(figsize=(14, 7))
sns.lineplot(data=grouped, x='year', y='rolling_avg', hue='species', marker='o')
plt.title('Trend Analysis of Salmon Escapement by Species (Year-wise, with 3-Year Rolling Average)')
plt.xlabel('Year')
plt.ylabel('Escapement (Rolling Avg of Abundance Quantity)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.show()

#Objective 2: Heatmap Visualization of Average Salmon Escapement by Species and Year
heatmap_df = df.copy()
heatmap_df = heatmap_df[['species', 'year', 'abundance_quantity']].dropna()
heatmap_df['year'] = heatmap_df['year'].astype(int)

pivot_table = heatmap_df.pivot_table(values='abundance_quantity', index='species', columns='year', aggfunc='mean')

plt.figure(figsize=(14, 8))
sns.heatmap(pivot_table, cmap='YlGnBu', linewidths=0.5, linecolor='gray')
plt.title('Heatmap of Average Salmon Escapement by Species and Year')
plt.xlabel('Year')
plt.ylabel('Species')
plt.tight_layout()
#plt.show()

#Objective 3: Total Escapement (Abundance) by Species (columns Chart using std )
species_data = df.copy()
species_data = species_data[(species_data['abundance_quantity'].notnull()) & (species_data['abundance_quantity'] > 0)]

species_total = species_data.groupby('species')['abundance_quantity'].std().reset_index()
species_total = species_total.sort_values(by='abundance_quantity', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=species_total, x='species', y='abundance_quantity', hue='species', dodge=False, legend=False)
plt.title("Standard Deviation of Escapement by Species")
plt.xlabel("Species")
plt.ylabel("Standard Deviation (Abundance Quantity)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()

#objective 4: Top 10 Sub-Populations by Median Salmon Escapement;

subpop_data = df.groupby('sub-population_name')['abundance_quantity'].median().reset_index()
top_10_subpops = subpop_data.sort_values(by='abundance_quantity', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_subpops, x='abundance_quantity', y='sub-population_name', hue='sub-population_name', legend=False, palette='mako')
plt.title("Top 10 Sub-Populations by Total Escapement")
plt.xlabel("Total Escapement")
plt.ylabel("Sub-Population Name")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
#plt.show()

#Objective 5: Correlation Analysis of Numerical Features Using Heatmap;
numeric_df = df.select_dtypes(include='number')
numeric_df = numeric_df.dropna()
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Fields")
plt.tight_layout()
#plt.show()

#Objective 6: Detect and Visualize Year-wise Outliers in Salmon Escapement Using Boxplot;
# Calculate outliers for each year
outliers = pd.DataFrame()

for year, group in df.groupby('year'):
    q1 = group['abundance_quantity'].quantile(0.25)
    q3 = group['abundance_quantity'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    year_outliers = group[(group['abundance_quantity'] < lower_bound) | (group['abundance_quantity'] > upper_bound)]
    outliers = pd.concat([outliers, year_outliers])

  
print(q1)
print(q3)
print(iqr)
print(lower_bound)
print(upper_bound)


plt.figure(figsize=(15, 7))
sns.boxplot(data=df, x='year', y='abundance_quantity')

plt.yscale('log')
plt.xticks(rotation=90)
plt.title('Year-wise Escapement with Outliers Highlighted')
plt.xlabel('Year')
plt.ylabel('Abundance Quantity (log scale)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
