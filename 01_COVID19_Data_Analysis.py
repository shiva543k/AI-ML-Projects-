import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv')

# Filter for a specific country
country = 'India'
df_country = df[df['Country'] == country]

# Plot COVID-19 cases over time
plt.figure(figsize=(10, 5))
plt.plot(df_country['Date'], df_country['Confirmed'], label='Confirmed Cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.title(f'COVID-19 Trend in {country}')
plt.legend()
plt.xticks(rotation=45)
plt.show()

