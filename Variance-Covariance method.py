import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# load data
file_path = "CAC40 data.xlsx"  
df = pd.read_excel(file_path, skiprows=2) 
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']  
df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')  
returns = df['Returns'].dropna() 
sorted_returns = returns.sort_values()

mu = returns.mean()
sigma = returns.std()

delta_t = 1
confidence_level_90 = 0.90
confidence_level_95 = 0.95
confidence_level_975 = 0.975
confidence_level_99 = 0.99
z_90 = norm.ppf(confidence_level_90)
z_95 = norm.ppf(confidence_level_95)
z_975 = norm.ppf(confidence_level_975)
z_99 = norm.ppf(confidence_level_99)

var_90 = mu + z_90 * sigma
var_95 = mu + z_95 * sigma
var_975 = mu + z_975 * sigma
var_99 = mu + z_99 * sigma

print(f"Variance-Covariance VaR (90%, 1-day): {var_90:.4%}")
print(f"Variance-Covariance VaR (95%, 1-day): {var_95:.4%}")
print(f"Variance-Covariance VaR (97.5%, 1-day): {var_975:.4%}")
print(f"Variance-Covariance VaR (99%, 1-day): {var_99:.4%}")

es_90 = -mu + sigma * norm.pdf(z_90) / (1 - confidence_level_90)
es_95 = -mu + sigma * norm.pdf(z_95) / (1 - confidence_level_95)
es_975 = -mu + sigma * norm.pdf(z_975) / (1 - confidence_level_975)
es_99 = -mu + sigma * norm.pdf(z_99) / (1 - confidence_level_99)


print(f"Variance-Covariance ES (90%, 1-day): {es_90:.4%}")
print(f"Variance-Covariance ES (95%, 1-day): {es_95:.4%}")
print(f"Variance-Covariance ES (97.5%, 1-day): {es_975:.4%}")
print(f"Variance-Covariance ES (99%, 1-day): {es_99:.4%}")

#plot
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, color='blue', lw=2, label='Normal PDF')

# VaR lines
plt.axvline(x=-var_90, color='black', linestyle='--', linewidth=2, label='VaR')
plt.text(-var_90, norm.pdf(-var_90, mu, sigma)*2, 'VaR(90%)', ha='right', va='bottom', rotation=90, fontsize=10)

plt.axvline(x=-var_95, color='black', linestyle='--', linewidth=2)
plt.text(-var_95, norm.pdf(-var_95, mu, sigma)*1.5, 'VaR(95%)', ha='right', va='bottom', rotation=90, fontsize=10)

plt.axvline(x=-var_99, color='black', linestyle='--', linewidth=2)
plt.text(-var_99, norm.pdf(-var_99, mu, sigma)*1, 'VaR(99%)', ha='right', va='bottom', rotation=90, fontsize=10)

# ES lines
plt.axvline(x=-es_90, color='red', linestyle=':', linewidth=2, label='ES')
plt.text(-es_90, norm.pdf(-es_90, mu, sigma)*0.1, 'ES(90%)', ha='right', va='bottom', rotation=90, fontsize=10, color='red')

plt.axvline(x=-es_95, color='red', linestyle=':', linewidth=2)
plt.text(-es_95, norm.pdf(-es_95, mu, sigma)*5.5, 'ES(95%)', ha='right', va='bottom', rotation=90, fontsize=10, color='red')

plt.axvline(x=-es_99, color='red', linestyle=':', linewidth=2)
plt.text(-es_99, norm.pdf(-es_99, mu, sigma)*20, 'ES(99%)', ha='right', va='bottom', rotation=90, fontsize=10, color='red')

plt.title('Normal Distribution of Daily Returns\nwith Variance-Covariance VaR and ES (90%, 95%, 99%)', fontsize=14)
plt.legend(loc='upper left', fontsize=10)

plt.xticks(np.arange(-0.1, 0.11, 0.02), [f'{x:.0%}' for x in np.arange(-0.1, 0.11, 0.02)])







