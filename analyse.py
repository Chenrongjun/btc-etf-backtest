import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Step 1: Data Sourcing
# Get Bitcoin price data from CoinGecko
# btc_price_url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365'
# btc_data = requests.get(btc_price_url, timeout=10).json()
# btc_prices = pd.DataFrame(btc_data['prices'], columns=['timestamp', 'price'])
# btc_prices['timestamp'] = pd.to_datetime(btc_prices['timestamp'], unit='ms')
# btc_prices.set_index('timestamp', inplace=True)
# btc_prices.to_hdf('data/btc_prices.h5',key='df', mode='w', complevel=1)
# Get ETF net flow data (manually from Farside)

btc_prices = pd.read_hdf('data/btc_prices.h5')
btc_prices['daily_return'] = btc_prices['price'].pct_change(1) # T日放置T-1到T的一日收益率
btc_prices['vol_20d'] = btc_prices['daily_return'].rolling(window=20).std() * np.sqrt(20) # T-20日到T日的波动率
btc_prices['vol_60d'] = btc_prices['daily_return'].rolling(window=60).std() * np.sqrt(60) # T-60日到T日的波动率
btc_prices['daily_return'] = btc_prices['daily_return'].shift(-1) # T日放置T到T+1的一日收益率
btc_prices['vol_20d'] = btc_prices['vol_20d'].shift(-20) # T日放置T到T+20的vol
btc_prices['vol_60d'] = btc_prices['vol_60d'].shift(-60) # T日放置T到T+60的vol

btc_prices = btc_prices.iloc[:-20]
etf_data = pd.read_csv('data/etf_net_flow.csv', parse_dates=['date'], index_col='date')
etf_data = etf_data.replace('-',np.nan).astype(float)

btc_prices.index = pd.to_datetime(btc_prices.index)
etf_data.index = pd.to_datetime(etf_data.index)
etf_data = etf_data.drop('BTC', axis=1)
# T日只能获取T-1的ETF信息
etf_data.index = pd.Series(etf_data.index).apply(lambda x:x+pd.Timedelta('1D'))

# 获取所有ETF列名（排除Total列）
etf_columns = [col for col in etf_data.columns if col != 'Total']

# Step 2: Merge Data
# ETF只在交易日披露，数量较少
merged_data = pd.merge(btc_prices, etf_data, left_index=True, right_index=True, how='inner')

# 创建结果存储字典
results = {}
correlation_matrix = {}
lag_correlations = {}

# 分析每个ETF与比特币价格的相关性
print("\n各ETF与比特币价格的相关系数:")
for etf in etf_columns + ['Total']:
    # 跳过包含太多缺失值的ETF
    if merged_data[etf].isna().sum() > len(merged_data) * 0.5:
        print(f"{etf}: 数据缺失过多，跳过分析")
        continue
    # 计算相关性
    valid_data = merged_data[['daily_return', etf]].dropna()
    correlation = valid_data['daily_return'].corr(valid_data[etf])
    correlation_matrix[etf] = correlation
    print(f"{etf}: {correlation:.4f}")
    # 创建滞后特征来检查领先/滞后关系
    lag_data = valid_data.copy()
    for i in range(1, 8):
        lag_data[f'{etf}_lag_{i}'] = lag_data[etf].shift(i)
        lag_data[f'ret_lag_{i}'] = lag_data['daily_return'].shift(i)
    # 删除NaN值
    lag_data = lag_data.dropna()
    # 计算滞后相关性
    etf_lag_corr = {}
    print(f"\n{etf}净流入领先比特币价格的相关性:")
    for i in range(1, 8):
        corr = lag_data['daily_return'].corr(lag_data[f'{etf}_lag_{i}'])
        etf_lag_corr[f'{etf}leading{i}天'] = corr
        print(f"{etf}leading{i}天: {corr:.4f}")
    print(f"\n比特币价格领先{etf}净流入的相关性:")
    for i in range(1, 8):
        corr = lag_data[etf].corr(lag_data[f'ret_lag_{i}'])
        etf_lag_corr[f'价格领先{etf}{i}天'] = corr
        print(f"价格领先{i}天: {corr:.4f}")
    lag_correlations[etf] = etf_lag_corr

# 可视化相关性矩阵
plt.figure(figsize=(10, 8))
corr_df = pd.DataFrame(list(correlation_matrix.items()), columns=['ETF', 'corr'])
corr_df = corr_df.sort_values('corr', ascending=False)

sns.barplot(x='corr', y='ETF', data=corr_df)
plt.tight_layout()
plt.savefig('etf_correlations.png')
plt.show()

# 详细可视化
etf = 'Total'
plt.figure(figsize=(16, 8))
# 价格和ETF流入的时间序列图
plt.subplot(1, 2, 1)
ax1 = plt.gca()
ax2 = ax1.twinx()
valid_data = merged_data[['daily_return', etf]].dropna()
ax1.plot(valid_data.index, valid_data['daily_return'], 'b-', label='btc ret')
ax2.plot(valid_data.index, valid_data[etf], 'r-', label=f'{etf} income')
ax1.set_xlabel('date')
ax1.set_ylabel('return', color='b')
ax2.set_ylabel(f'{etf} income', color='r')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
# 滞后相关性图
plt.subplot(1, 2, 2)
lag_df = pd.DataFrame(list(lag_correlations[etf].items()), columns=['lag', 'corr'])
sns.barplot(x='lag', y='corr', data=lag_df)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{etf}_analysis.png')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.show()

# 分析ETF与比特币vol的相关性
valid_data = merged_data[['vol_20d', etf]].dropna()
correlation = valid_data['vol_20d'].corr(valid_data[etf])
print(f"{correlation:.4f}")
valid_data = merged_data[['vol_60d', etf]].dropna()
correlation = valid_data['vol_60d'].corr(valid_data[etf])
print(f"{correlation:.4f}")

