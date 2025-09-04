import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2

file_path = "CAC40 data.xlsx"
df = pd.read_excel(file_path, skiprows=2)
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']
df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')
returns = df['Returns'].dropna().reset_index(drop=True)

window = 180
alpha=0.025
alpha90 = 0.10          
alpha95 = 0.05 
alpha975 = 0.025         
alpha99 = 0.01
holding_period = 1    
np.random.seed(8888)


def kupiec_pof_test(violations, alpha):
    n = len(violations)
    x = violations.sum()  
    pi = x / n 
    pi0 = 1 - alpha 

    if pi == 0 or pi == 1:
        return float('nan'), 1.0  

    LR_pof = -2 * (np.log((1 - pi0)**(n - x) * pi0**x) -
                   np.log((1 - pi)**(n - x) * pi**x))
    p_value = 1 - chi2.cdf(LR_pof, df=1)

    return LR_pof, p_value
############################## Historical VaR #################################
historical_var = []
actual_return = []
violations_hist = []

for i in range(window, len(returns)):
    rolling_data = returns[i - window:i]
    var = np.percentile(rolling_data, alpha * 100) 
    actual = returns[i]                              

    historical_var.append(var)
    actual_return.append(actual)
    violations_hist.append(actual < var)

violations_hist = np.array(violations_hist)
actual_return = np.array(actual_return)
historical_var = np.array(historical_var)

# Add Kupiec POF Test
LR_pof_hist, pval_hist = kupiec_pof_test(violations_hist, 1 - alpha)

print(f"Historical VaR")
print(f"Total test days: {len(violations_hist)}")
print(f"Number of VaR violations: {violations_hist.sum()}")
print(f"Violation rate: {violations_hist.mean():.2%} (Expected: {alpha:.2%})")
print(f"Kupiec POF Test (Historical VaR):")
print(f"LR_pof statistic: {LR_pof_hist:.4f}, p-value: {pval_hist:.4f}")


plt.figure(figsize=(14, 6))
plt.plot(actual_return, label="Actual Daily Return", color='blue')
plt.plot(historical_var, label=f"Historical VaR({int((1 - alpha) * 100)}%)", color='red')
plt.scatter(np.where(violations_hist)[0], actual_return[violations_hist],
            color='orange', label='Violation', zorder=5)
plt.title("Backtesting: Historical VaR vs Actual Return")
plt.xlabel("Days")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

############################## Monte Carlo VaR #################################

n_simulations = 10000
mc_var = []
violations_mc = [] 

for i in range(window, len(returns)):
    rolling_data = returns[i - window:i]
    mu = rolling_data.mean()
    sigma = rolling_data.std()

    simulated_returns = np.random.normal(loc=mu * holding_period,
                                         scale=sigma * np.sqrt(holding_period),
                                         size=n_simulations)

    var = np.percentile(simulated_returns, alpha * 100)
    actual = returns[i]

    mc_var.append(var)
    violations_mc.append(actual < var)

violations_mc = np.array(violations_mc)
mc_var = np.array(mc_var)

LR_pof_mc, pval_mc = kupiec_pof_test(violations_mc, 1 - alpha)

print(f"Monte Carlo VaR")
print(f"Total test days: {len(violations_mc)}")
print(f"Number of VaR violations: {violations_mc.sum()}")
print(f"Violation rate: {violations_mc.mean():.2%} (Expected: {alpha:.2%})")
print(f"Kupiec POF Test (Monte Carlo VaR):")
print(f"LR_pof statistic: {LR_pof_mc:.4f}, p-value: {pval_mc:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(actual_return, label="Actual Daily Return", color='blue')
plt.plot(mc_var, label=f"Monte Carlo VaR({int((1 - alpha)*100)}%)", color='green')
plt.scatter(np.where(violations_mc)[0], actual_return[violations_mc],
            color='orange', label='Violation', zorder=5)
plt.title("Backtesting: Monte Carlo VaR vs Actual Return")
plt.xlabel("Days")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


############################## Variance-Covariance VaR #################################

z_score95 = norm.ppf(1 - alpha95)
z_score99 = norm.ppf(1 - alpha99)
z_score = norm.ppf(1 - alpha)

vcv_var = []
violations_vcv = []

for i in range(window, len(returns)):
    rolling_data = returns[i - window:i]
    mu = rolling_data.mean()
    sigma = rolling_data.std()

    var = mu * holding_period - z_score * sigma * np.sqrt(holding_period)
    actual = returns[i]

    vcv_var.append(var)
    violations_vcv.append(actual < var)

violations_vcv = np.array(violations_vcv)
vcv_var = np.array(vcv_var)

LR_pof_vcv, pval_vcv = kupiec_pof_test(violations_vcv, 1 - alpha)

print(f"Variance-Covariance VaR")
print(f"Total test days: {len(violations_vcv)}")
print(f"Number of VaR violations: {violations_vcv.sum()}")
print(f"Violation rate: {violations_vcv.mean():.2%} (Expected: {alpha:.2%})")
print(f"Kupiec POF Test (VCV VaR):")
print(f"LR_pof statistic: {LR_pof_vcv:.4f}, p-value: {pval_vcv:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(actual_return, label="Actual Daily Return", color='blue', linewidth=0.8)
plt.plot(vcv_var, label=f"VCV VaR({int((1 - alpha)*100)}%)", color='purple')
plt.scatter(np.where(violations_vcv)[0], actual_return[violations_vcv],
            color='orange', label='VCV Violation', zorder=5)
plt.title("Backtesting: Variance-Covariance VaR (99%)")
plt.xlabel("Days")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

############################## Comparison #################################

plt.figure(figsize=(15, 6))
plt.plot(-actual_return, label='Actual Loss', color='black', linewidth=1)
plt.plot(-historical_var, label='Historical VaR(97.5%)', color='blue')
plt.plot(-mc_var, label='Monte Carlo VaR(97.5%)', color='orange')
plt.plot(-vcv_var, label='Variance-Covariance VaR(97.5%)', color='purple', linestyle='--')

plt.scatter(np.where(violations_hist)[0], -actual_return[violations_hist],
            color='purple', marker='x', label='Hist Violation', zorder=10)
plt.scatter(np.where(violations_mc)[0], -actual_return[violations_mc],
            color='orange', marker='*', label='MC Violation', zorder=3)
plt.scatter(np.where(violations_vcv)[0], -actual_return[violations_vcv],
            color='red', marker='P', label='VCV Violation')

plt.title("Backtesting Comparison: Actual Loss vs VaR (Historical, Monte Carlo, Variance-Covariance)", fontsize=20)
plt.xlabel("Days")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("Hist violations:", np.where(violations_hist)[0])
print("MC violations:", np.where(violations_mc)[0])
print("VCV violations:", np.where(violations_vcv)[0])




