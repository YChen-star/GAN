import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# load data
file_path = "CAC40 data.xlsx"
df = pd.read_excel(file_path, skiprows=2)
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']
df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')
returns = df['Returns'].dropna()

# parameter
alpha_995 = 0.995
alpha_95 = 0.95
alpha_975 = 0.975
alpha_99 = 0.99

losses = -returns
threshold_u = np.percentile(losses, 94) 
excess_losses = losses[losses > threshold_u] - threshold_u

params = genpareto.fit(excess_losses, floc=0)
xi, loc, beta = params

N = len(losses)
k = len(excess_losses)
var_evt995 = threshold_u + (beta / xi) * (((N / k) * (1 - alpha_995))**(-xi) - 1)
var_evt95 = threshold_u + (beta / xi) * (((N / k) * (1 - alpha_95))**(-xi) - 1)
var_evt975 = threshold_u + (beta / xi) * (((N / k) * (1 - alpha_975))**(-xi) - 1)
var_evt99 = threshold_u + (beta / xi) * (((N / k) * (1 - alpha_99))**(-xi) - 1)

es_evt995 = (var_evt995 + (beta - xi * threshold_u)) / (1 - xi)
es_evt95 = (var_evt95 + (beta - xi * threshold_u)) / (1 - xi)
es_evt975 = (var_evt975 + (beta - xi * threshold_u)) / (1 - xi)
es_evt99 = (var_evt99 + (beta - xi * threshold_u)) / (1 - xi)

print(f"EVT-based VaR (95%): {var_evt95:.4%}")
print(f"EVT-based ES  (95%): {es_evt95:.4%}")

print(f"EVT-based VaR (99%): {var_evt99:.4%}")
print(f"EVT-based ES  (99%): {es_evt99:.4%}")

print(f"EVT-based VaR (97.5%): {var_evt975:.4%}")
print(f"EVT-based ES  (97.5%): {es_evt975:.4%}")

print(f"EVT-based VaR (99.5%): {var_evt995:.4%}")
print(f"EVT-based ES  (99.5%): {es_evt995:.4%}")


# Plotting
import seaborn as sns
sns.histplot(excess_losses, bins=30, stat="density", label="Excess Losses", color='skyblue')
x_vals = np.linspace(min(excess_losses), max(excess_losses), 100)
plt.plot(x_vals, genpareto.pdf(x_vals, *params), 'r-', label='GPD Fit')
plt.title('GPD Fit on Excess Losses (u = 95% Quantile)')
plt.xlabel('Excess over Threshold')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
