import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

file_path = "CAC40 data.xlsx"  
df = pd.read_excel(file_path, skiprows=2) 
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']  
df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')  
returns = df['Returns'].dropna() 
sorted_returns = returns.sort_values()

mu = returns.mean()
sigma = returns.std()

delta_t = 1
confidence_level_95 = 0.95
confidence_level_99 = 0.99
confidence_level_975 = 0.975
z_95 = norm.ppf(1 - confidence_level_95)
z_99 = norm.ppf(1 - confidence_level_99)
z_975 = norm.ppf(1 - confidence_level_975)

vc_var_95 = -(mu + z_95 * sigma)
vc_var_99 = -(mu + z_99 * sigma)
vc_var_975 = -(mu + z_975 * sigma)

var_95 = mu + z_95 * sigma
var_99 = mu + z_99 * sigma
var_975 = mu + z_975 * sigma


# print result
print(f"Variance-Covariance VaR (95%, 1-day): {vc_var_95:.4%}")
print(f"Variance-Covariance VaR (99%, 1-day): {vc_var_99:.4%}")
print(f"Variance-Covariance VaR (97.5%, 1-day): {vc_var_975:.4%}")

#plot
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, color='blue', lw=2)

# point VaR(95%) and VaR(99%)
plt.axvline(x=var_95, color='black', linestyle='--', linewidth=2)
plt.text(var_95, norm.pdf(var_95, mu, sigma)*1.05, 'VaR(95%)', ha='right', va='bottom', rotation=90, fontsize=10)

plt.axvline(x=var_99, color='black', linestyle='--', linewidth=2)
plt.text(var_99, norm.pdf(var_99, mu, sigma)*1.05, 'VaR(99%)', ha='right', va='bottom', rotation=90, fontsize=10)

plt.title('Normal Distribution of Returns with VaR(95%) and VaR(99%)')
plt.xlabel('Returns')
plt.ylabel('Probability density')
plt.xticks(np.arange(-0.1, 0.11, 0.02), [f'{x:.0%}' for x in np.arange(-0.1, 0.11, 0.02)])
plt.yticks(np.linspace(0, np.max(pdf)*1.2, 7), [f'{x:.0f}' for x in np.linspace(0, np.max(pdf)*1.2, 7)])

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
