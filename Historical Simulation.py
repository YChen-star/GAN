import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, PercentFormatter


file_path = "CAC40 data.xlsx"  
df = pd.read_excel(file_path, skiprows=2)  
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']  
df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')  
returns = df['Returns'].dropna() 
sorted_returns = returns.sort_values()


confidence_level90 = 0.90
alpha90 = 1 - confidence_level90


confidence_level95 = 0.95
alpha95 = 1 - confidence_level95

confidence_level975 = 0.975
alpha975 = 1 - confidence_level975

confidence_level99 = 0.99
alpha99 = 1 - confidence_level99


var90 = -np.percentile(returns, alpha90 * 100)
var95 = -np.percentile(returns, alpha95 * 100)
var975 = -np.percentile(returns, alpha975 * 100)
var99 = -np.percentile(returns, alpha99 * 100)

historical_var90 = np.percentile(returns, alpha90 * 100)
historical_var95 = np.percentile(returns, alpha95 * 100)
historical_var975 = np.percentile(returns, alpha975 * 100)
historical_var99 = np.percentile(returns, alpha99 * 100)

es90 = -returns[returns <= -var90].mean()
es95 = -returns[returns <= -var95].mean()
es975 = -returns[returns <= -var975].mean()
es99 = -returns[returns <= -var99].mean()

plt.figure(figsize=(10, 6))
sns.histplot(returns, bins=30, stat="probability", color='skyblue', edgecolor='black')

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.01))           
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

# Add VaR lines
plt.axvline(x=historical_var90, color='black', linestyle='--', linewidth=2)
plt.text(historical_var90, plt.ylim()[1]*0.8, 'VaR(90%)', rotation=90, verticalalignment='center', fontsize=10)


plt.axvline(x=historical_var95, color='black', linestyle='--', linewidth=2)
plt.text(historical_var95, plt.ylim()[1]*0.5, 'VaR(95%)', rotation=90, verticalalignment='center', fontsize=10)

plt.axvline(x=historical_var99, color='black', linestyle='--', linewidth=2)
plt.text(historical_var99, plt.ylim()[1]*0.2, 'VaR(99%)', rotation=90, verticalalignment='center', fontsize=10)

# Add ES lines
plt.axvline(x=-es90, color='red', linestyle=':', linewidth=2)
plt.text(-es90, plt.ylim()[1]*0.8, 'ES(90%)', rotation=90, verticalalignment='center', fontsize=10, color='red')

plt.axvline(x=-es95, color='red', linestyle=':', linewidth=2)
plt.text(-es95, plt.ylim()[1]*0.5, 'ES(95%)', rotation=90, verticalalignment='center', fontsize=10, color='red')

plt.axvline(x=-es99, color='red', linestyle=':', linewidth=2)
plt.text(-es99, plt.ylim()[1]*0.2, 'ES(99%)', rotation=90, verticalalignment='center', fontsize=10, color='red')

plt.title('Histogram of Daily Returns with VaR(90%), VaR(95%) and VaR(99%), ES(90%), ES(95%) and ES(99%)', fontsize=14)
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"Historical Simulation VaR (90%, 1-day): {var90:.4%}")
print(f"Historical Simulation VaR (95%, 1-day): {var95:.4%}")
print(f"Historical Simulation VaR (97.5%, 1-day): {var975:.4%}")
print(f"Historical Simulation VaR (99%, 1-day): {var99:.4%}")

print(f"Historical Simulation ES (90%, 1-day): {es90:.4%}")
print(f"Historical Simulation ES (95%, 1-day): {es95:.4%}")
print(f"Historical Simulation ES (97.5%, 1-day): {es975:.4%}")
print(f"Historical Simulation ES (99%, 1-day): {es99:.4%}")


portfolio_value = 1_000_000
es95_euro = es95 * portfolio_value
print(f"Estimated 1-day ES (95%) loss in €: €{es95_euro:,.2f}")
