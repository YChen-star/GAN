import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt
from scipy.stats import genpareto

file_path = "CAC40 data.xlsx"
df = pd.read_excel(file_path, skiprows=2)
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']

returns = pd.to_numeric(df['Returns'], errors='coerce').dropna().reset_index(drop=True)
losses = -returns  

window = 180
alpha = 0.025
pot_percentile = 100 * (1 - alpha)
evt_threshold_percentile = 94  

hs_var_list, hs_es_list = [], []
vcv_var_list, vcv_es_list = [], []
evt_var_list, evt_es_list = [], []
actual_loss_list = []

for i in range(window, len(losses)):
    roll = losses[i-window:i]
    mu, sigma = roll.mean(), roll.std(ddof=1)
    z = norm.ppf(1 - alpha)


    hs_var = np.percentile(roll, 100 * (1 - alpha))
    hs_es = roll[roll >= hs_var].mean()
    hs_var_list.append(hs_var)
    hs_es_list.append(hs_es)

    vcv_var = mu + z * sigma
    vcv_es = mu + sigma * norm.pdf(z) / alpha
    vcv_var_list.append(vcv_var)
    vcv_es_list.append(vcv_es)

    u = np.percentile(roll, evt_threshold_percentile)
    exceed = roll[roll > u] - u
    nu = len(exceed)
    n = len(roll)

    if nu > 10:
        try:
            c, loc, scale = genpareto.fit(exceed, floc=0)
            if (scale > 0) and (c < 1)and (np.abs(c) < 0.7):
                q = (n / nu) * alpha
                if c != 0:
                    var_evt = u + (scale / c) * (q ** (-c) - 1)
                else:
                    var_evt = u + scale * np.log(1 / q)

                es_evt = (var_evt + (scale - c * u) / (1 - c)) / (1 - alpha)
            else:
                var_evt, es_evt = np.nan, np.nan
        except Exception as e:
            var_evt, es_evt = np.nan, np.nan
    else:
        var_evt, es_evt = np.nan, np.nan
    evt_var_list.append(var_evt)
    evt_es_list.append(es_evt)

    actual_loss_list.append(losses[i])

hs_var_arr, hs_es_arr = np.array(hs_var_list), np.array(hs_es_list)
vcv_var_arr, vcv_es_arr = np.array(vcv_var_list), np.array(vcv_es_list)
evt_var_arr, evt_es_arr = np.array(evt_var_list), np.array(evt_es_list)
loss_arr = np.array(actual_loss_list)
dates = df['Date'].iloc[window:len(losses)].reset_index(drop=True)

viol_hs_var = (loss_arr > hs_var_arr)
viol_hs_es = (loss_arr > hs_es_arr)
viol_vcv_var = (loss_arr > vcv_var_arr)
viol_vcv_es = (loss_arr > vcv_es_arr)
viol_evt_es = (loss_arr > evt_es_arr) & np.isfinite(evt_es_arr)

valid_evt = np.isfinite(evt_es_arr)
print(f'HS ES effective window ratio: 100.0%')
print(f'VCV ES effective window ratio: 100.0%')
print(f'EVT ES effective window ratio: {valid_evt.mean() * 100:.1f}%')
print(f'HS ES violation points: {viol_hs_es.sum()}，ratio: {viol_hs_es.mean():.2%}')
print(f'VCV ES violation points: {viol_vcv_es.sum()}，ratio: {viol_vcv_es.mean():.2%}')
print(f'EVT ES violation points: {viol_evt_es.sum()}，ratio: {viol_evt_es[valid_evt].mean():.2%} (Only within the valid window)')

# 6. Acerbi–Szekely backtest
def acerbi_szekely_test(loss, var, es, alpha):
    indicator = (loss > var).astype(int)
    S = (1/alpha) * indicator * (loss - var) - (es - var)
    S = S[np.isfinite(S)]
    S_mean = np.mean(S)
    S_std = np.std(S, ddof=1)
    T = len(S)
    if T == 0:
        return np.nan, np.nan, np.nan
    t_stat = S_mean / (S_std / np.sqrt(T))
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=T-1))
    return t_stat, p_value, S_mean

print("=== Historical Simulation (HS) ES ===")
t_stat_hs, p_value_hs, S_mean_hs = acerbi_szekely_test(loss_arr, hs_var_arr, hs_es_arr, alpha)
print(f'HS ES AS t statistic: {t_stat_hs:.4f}')
print(f'HS ES p-value: {p_value_hs:.4f}')
print(f'HS ES Smean: {S_mean_hs:.6f}\n')

print("=== Variance-Covariance (VCV) ES ===")
t_stat_vcv, p_value_vcv, S_mean_vcv = acerbi_szekely_test(loss_arr, vcv_var_arr, vcv_es_arr, alpha)
print(f'VCV ES AS t statistic: {t_stat_vcv:.4f}')
print(f'VCV ES p-value: {p_value_vcv:.4f}')
print(f'VCV ES Smean: {S_mean_vcv:.6f}')

print("=== EVT (POT) ES ===")
t_stat_evt, p_value_evt, S_mean_evt = acerbi_szekely_test(loss_arr[valid_evt], evt_var_arr[valid_evt], evt_es_arr[valid_evt], alpha)
print(f'EVT ES AS t statistic: {t_stat_evt:.4f}')
print(f'EVT ES p-value: {p_value_evt:.4f}')
print(f'EVT ES Smean: {S_mean_evt:.6f}')

plt.figure(figsize=(15, 6))
x = range(len(loss_arr))

plt.plot(x, loss_arr, label='Actual Loss', color='black', lw=1)
plt.plot(x, hs_es_arr, label='Historical ES (97.5%)', color='red', lw=2)
plt.plot(x, vcv_es_arr, label='Variance-Covariance ES (97.5%)', color='green', lw=2)
plt.plot(x, evt_es_arr, label='EVT-based ES (97.5%)', color='orange', lw=2)
plt.scatter(np.where(viol_hs_es)[0], loss_arr[viol_hs_es], color='red', label='Hist Violation', zorder=5, marker='o')
plt.scatter(np.where(viol_vcv_es)[0], loss_arr[viol_vcv_es], color='cyan', label='VCV Violation', zorder=5, marker='x')
plt.scatter(np.where(viol_evt_es)[0], loss_arr[viol_evt_es], color='magenta', label='EVT Violation', zorder=5, marker='d')
plt.title('Backtesting Comparison: Actual Loss vs ES (Historical, Variance-Covariance, EVT)', fontsize=20)
plt.ylabel('Loss')
plt.xlabel('Date')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


print("===== Summary Statistics for ES Estimates (97.5%) =====")
print(f"HS ES:   mean={np.nanmean(hs_es_arr):.6f}, median={np.nanmedian(hs_es_arr):.6f}, std={np.nanstd(hs_es_arr):.6f}")
print(f"VCV ES:  mean={np.nanmean(vcv_es_arr):.6f}, median={np.nanmedian(vcv_es_arr):.6f}, std={np.nanstd(vcv_es_arr):.6f}")
print(f"EVT ES:  mean={np.nanmean(evt_es_arr[valid_evt]):.6f}, median={np.nanmedian(evt_es_arr[valid_evt]):.6f}, std={np.nanstd(evt_es_arr[valid_evt]):.6f}")
print()