import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "CAC40 data.xlsx"
df = pd.read_excel(file_path, skiprows=2)
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']
df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')
returns = df['Returns'].dropna()

mu = returns.mean()
sigma = returns.std()
n_simulations = 10000  
holding_days = 1       
alpha_95 = 0.05
alpha_99 = 0.01
alpha_975 = 0.025

simulated_returns = np.random.normal(loc=mu * holding_days,
                                     scale=sigma * np.sqrt(holding_days),
                                     size=n_simulations)

# calculate VaR
mc_var_95 = -np.percentile(simulated_returns, alpha_95 * 100)
mc_var_99 = -np.percentile(simulated_returns, alpha_99 * 100)
mc_var_975 = -np.percentile(simulated_returns, alpha_975 * 100)

# print result
print(f"Monte Carlo VaR (95%, 1-day): {mc_var_95:.4%}")
print(f"Monte Carlo VaR (99%, 1-day): {mc_var_99:.4%}")
print(f"Monte Carlo VaR (97.5%, 1-day): {mc_var_975:.4%}")

# plot
plt.figure(figsize=(10, 6))
plt.hist(simulated_returns, bins=50, color='lightblue', edgecolor='black', density=True)
plt.axvline(-mc_var_95, color='black', linestyle='--', label='VaR(95%)')
plt.axvline(-mc_var_99, color='black', linestyle='--', label='VaR(99%)')
plt.xlabel("Simulated Returns")
plt.ylabel("Probability Density")
plt.title("Monte Carlo Simulated Returns with VaR(95%) and VaR(99%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
