import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2


file_path = "generated_returns.csv"   
df = pd.read_csv(file_path)
returns = df["Generated_Return"].dropna().reset_index(drop=True)
losses = -returns 

window = 180
alpha = 0.025         
n_simulations = 10000
holding_period = 1
np.random.seed(8888)

def kupiec_pof_test(violations, alpha):
    n = len(violations)
    x = violations.sum()
    pi = x / n
    pi0 = alpha
    if x == 0 or x == n:
        return float('nan'), 1.0
    logL0 = (n - x) * np.log(1 - pi0) + x * np.log(pi0)
    logL1 = (n - x) * np.log(1 - pi) + x * np.log(pi)
    LR_pof = -2 * (logL0 - logL1)
    p_value = 1 - chi2.cdf(LR_pof, df=1)
    return LR_pof, p_value



# ---------- HS VaR ----------
historical_var, actual_loss, violations_hist = [], [], []
for i in range(window, len(losses)):
    rolling_data = losses[i-window:i]
    var = np.percentile(rolling_data, 100*(1-alpha))
    actual = losses[i]
    historical_var.append(var)
    actual_loss.append(actual)
    violations_hist.append(actual > var)
historical_var = np.array(historical_var)
actual_loss = np.array(actual_loss)
violations_hist = np.array(violations_hist)
LR_pof_hist, pval_hist = kupiec_pof_test(violations_hist, alpha)

# ---------- Monte Carlo VaR ----------
mc_var, violations_mc = [], []
for i in range(window, len(losses)):
    rolling_data = losses[i-window:i]
    mu = rolling_data.mean()
    sigma = rolling_data.std()
    simulated_losses = np.random.normal(loc=mu*holding_period, scale=sigma*np.sqrt(holding_period), size=n_simulations)
    var = np.percentile(simulated_losses, 100*(1-alpha))
    actual = losses[i]
    mc_var.append(var)
    violations_mc.append(actual > var)
mc_var = np.array(mc_var)
violations_mc = np.array(violations_mc)
LR_pof_mc, pval_mc = kupiec_pof_test(violations_mc, alpha)

# ---------- Variance-Covariance VaR ----------
z_score = norm.ppf(1 - alpha)
vcv_var, violations_vcv = [], []
for i in range(window, len(losses)):
    rolling_data = losses[i-window:i]
    mu = rolling_data.mean()
    sigma = rolling_data.std()
    var = mu*holding_period + z_score*sigma*np.sqrt(holding_period)
    actual = losses[i]
    vcv_var.append(var)
    violations_vcv.append(actual > var)
vcv_var = np.array(vcv_var)
violations_vcv = np.array(violations_vcv)
LR_pof_vcv, pval_vcv = kupiec_pof_test(violations_vcv, alpha)

# ---------- Output ----------
print("Historical VaR")
print(f"  Test days: {len(violations_hist)}")
print(f"  VaR violations: {violations_hist.sum()} ({violations_hist.mean():.2%}) (Expected: {alpha:.2%})")
print(f"  Kupiec POF: LR={LR_pof_hist:.4f}, p={pval_hist:.4f}")

print("Monte Carlo VaR")
print(f"  VaR violations: {violations_mc.sum()} ({violations_mc.mean():.2%})")
print(f"  Kupiec POF: LR={LR_pof_mc:.4f}, p={pval_mc:.4f}")

print("Variance-Covariance VaR")
print(f"  VaR violations: {violations_vcv.sum()} ({violations_vcv.mean():.2%})")
print(f"  Kupiec POF: LR={LR_pof_vcv:.4f}, p={pval_vcv:.4f}")

# ---------- Plot ----------
plt.figure(figsize=(15, 6))
plt.plot(actual_loss, label='generayed Loss', color='black', linewidth=1)
plt.plot(historical_var, label='Historical VaR(97.5%)', color='blue')
plt.plot(mc_var, label='Monte Carlo VaR(97.5%)', color='orange')
plt.plot(vcv_var, label='Variance-Covariance VaR(97.5%)', color='purple', linestyle='--')
plt.scatter(np.where(violations_hist)[0], actual_loss[violations_hist], color='blue', marker='x', label='Hist Violation', zorder=10)
plt.scatter(np.where(violations_mc)[0], actual_loss[violations_mc], color='orange', marker='*', label='MC Violation', zorder=3)
plt.scatter(np.where(violations_vcv)[0], actual_loss[violations_vcv], color='red', marker='P', label='VCV Violation')
plt.title("Backtesting Comparison: Generated Loss vs VaR (Historical, Monte Carlo, Variance-Covariance)", fontsize=20)
plt.xlabel("Days")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("===== Summary Statistics for VaR Estimates (97.5%) =====")
print(f"HS VaR:   mean={np.nanmean(historical_var):.6f}, median={np.nanmedian(historical_var):.6f}, std={np.nanstd(historical_var):.6f}")
print(f"VCV VaR:  mean={np.nanmean(vcv_var):.6f}, median={np.nanmedian(vcv_var):.6f}, std={np.nanstd(vcv_var):.6f}")
print(f"MC VaR:  mean={np.nanmean(mc_var):.6f}, median={np.nanmedian(mc_var):.6f}, std={np.nanstd(mc_var):.6f}")
print()




