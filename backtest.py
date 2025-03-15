import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = fm.FontProperties(fname=font_path)

def run_backtest(signals, initial_capital=100000, stop_loss_pct=-0.10, take_profit_pct=0.20):
    # 初始化结果数据
    results = signals.copy()
    results['position'] = 0
    results['portfolio_value'] = initial_capital
    results['btc_units'] = 0
    results['trade_type'] = ''
    results['trade_price'] = np.nan
    results['strategy_return'] = 0
    # 初始化交易状态变量
    current_capital = initial_capital
    entry_price = 0  # 入场价格
    trades = []  # 交易记录
    # 执行回测
    for i in range(1, len(results)):
        date = results.index[i]
        prev_date = results.index[i - 1]
        price = results.loc[date, 'price']
        signal = results.loc[date, 'signal']
        # 默认继承前一天的持仓状态
        results.loc[date, 'position'] = results.loc[prev_date, 'position']
        results.loc[date, 'btc_units'] = results.loc[prev_date, 'btc_units']
        # 风险管理: 检查止损和止盈条件
        if results.loc[prev_date, 'position'] == 1:
            price_change = (price - entry_price) / entry_price
            # 止损条件: 价格下跌超过止损比例
            if price_change < stop_loss_pct:
                # 执行止损
                btc_value = results.loc[prev_date, 'btc_units'] * price
                current_capital = btc_value
                results.loc[date, 'position'] = 0
                results.loc[date, 'btc_units'] = 0
                results.loc[date, 'trade_type'] = '止损'
                results.loc[date, 'trade_price'] = price
                trades.append({
                    'date': date,
                    'type': '止损',
                    'price': price,
                    'units': results.loc[prev_date, 'btc_units'],
                    'value': btc_value
                })
            # 止盈条件: 价格上涨超过止盈比例
            elif price_change > take_profit_pct:
                # 执行止盈
                btc_value = results.loc[prev_date, 'btc_units'] * price
                current_capital = btc_value
                results.loc[date, 'position'] = 0
                results.loc[date, 'btc_units'] = 0
                results.loc[date, 'trade_type'] = '止盈'
                results.loc[date, 'trade_price'] = price
                trades.append({
                    'date': date,
                    'type': '止盈',
                    'price': price,
                    'units': results.loc[prev_date, 'btc_units'],
                    'value': btc_value
                })
        # 交易信号处理
        # 做多信号: 当前空仓且IBIT资金净流出
        if signal == 1 and results.loc[prev_date, 'position'] == 0:
            # 买入比特币
            btc_units = current_capital / price
            results.loc[date, 'position'] = 1
            results.loc[date, 'btc_units'] = btc_units
            results.loc[date, 'trade_type'] = '买入'
            results.loc[date, 'trade_price'] = price
            entry_price = price
            trades.append({
                'date': date,
                'type': '买入',
                'price': price,
                'units': btc_units,
                'value': current_capital
            })
        # 平仓信号: 当前持仓且IBIT资金净流入
        elif signal == -1 and results.loc[prev_date, 'position'] == 1:
            # 卖出比特币
            btc_value = results.loc[prev_date, 'btc_units'] * price
            current_capital = btc_value
            results.loc[date, 'position'] = 0
            results.loc[date, 'btc_units'] = 0
            results.loc[date, 'trade_type'] = '卖出'
            results.loc[date, 'trade_price'] = price
            trades.append({
                'date': date,
                'type': '卖出',
                'price': price,
                'units': results.loc[prev_date, 'btc_units'],
                'value': btc_value
            })
        # 计算当日组合价值
        if results.loc[date, 'position'] == 1:
            # 持有比特币
            portfolio_value = results.loc[date, 'btc_units'] * price
        else:
            # 空仓，持有现金
            portfolio_value = current_capital
        results.loc[date, 'portfolio_value'] = portfolio_value
        # 计算策略收益率
        if i > 0:
            prev_value = results.loc[prev_date, 'portfolio_value']
            if prev_value > 0:
                results.loc[date, 'strategy_return'] = (portfolio_value - prev_value) / prev_value
    # 计算累积收益
    results['cumulative_return'] = (1 + results['strategy_return']).cumprod() - 1
    # 计算买入持有策略的收益
    initial_price = results['price'].iloc[0]
    results['buy_hold_units'] = initial_capital / initial_price
    results['buy_hold_value'] = results['buy_hold_units'] * results['price']
    results['buy_hold_return'] = results['buy_hold_value'].pct_change()
    results['buy_hold_cumulative_return'] = (1 + results['buy_hold_return']).cumprod() - 1
    return results, trades

# 计算性能指标函数
def calculate_metrics(results, trades, daily_return):
    # 1. 累计回报
    final_return = results['cumulative_return'].iloc[-1]
    buy_hold_return = results['buy_hold_cumulative_return'].iloc[-1]
    # 2. 年化收益率 (假设252个交易日)
    n_days = len(results)
    annual_return = (1 + final_return) ** (252 / n_days) - 1
    buy_hold_annual_return = (1 + buy_hold_return) ** (252 / n_days) - 1
    # 3. 夏普比率 (假设无风险利率为0)
    daily_returns = results['strategy_return'].dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    buy_hold_daily_returns = results['buy_hold_return'].dropna()
    buy_hold_sharpe = np.sqrt(252) * buy_hold_daily_returns.mean() / buy_hold_daily_returns.std() if buy_hold_daily_returns.std() > 0 else 0
    # 4. 最大回撤
    cumulative_returns = results['cumulative_return']+1.0  # 收益+1.0净值化计算
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    buy_hold_cumulative = results['buy_hold_cumulative_return']+1.0
    buy_hold_running_max = buy_hold_cumulative.cummax()
    buy_hold_drawdown = (buy_hold_cumulative - buy_hold_running_max) / buy_hold_running_max
    buy_hold_max_drawdown = buy_hold_drawdown.min()
    # 5. 胜率
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        # 计算每笔交易的盈亏
        trades_df['ret'] = daily_return.reindex(trades_df.date).values
        flag = 0
        for i in range(len(trades_df)):
            if trades_df.iloc[i]['type'] in ['卖出', '止盈', '止损'] and trades_df.iloc[i]['ret']<0:
                flag += 1
            elif trades_df.iloc[i]['type'] in ['买入'] and trades_df.iloc[i]['ret']>0:
                flag += 1
        # 计算胜率
        win_rate = flag/trades_df.shape[0]
    else:
        win_rate = 0
    # 6. 交易次数
    n_trades = len(trades_df)
    metrics = {
        '累计回报': final_return,
        '年化收益率': annual_return,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '胜率': win_rate,
        '交易次数': n_trades,
        '买入持有累计回报': buy_hold_return,
        '买入持有年化收益率': buy_hold_annual_return,
        '买入持有夏普比率': buy_hold_sharpe,
        '买入持有最大回撤': buy_hold_max_drawdown
    }
    return metrics


# 可视化回测结果函数
def plot_results(results, trades, metrics, figsize=(15, 9)):
    # 创建子图
    fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 2]})
    # 1. 价格和组合价值图
    ax1 = axes[0]
    ax1.set_title('value', fontsize=14)
    ax1.plot(results.index, results['price'], 'gray', alpha=0.5, label='price')
    ax1.set_ylabel('USD', fontsize=12)
    # 添加第二个Y轴显示组合价值
    ax1_2 = ax1.twinx()
    ax1_2.plot(results.index, results['portfolio_value'], 'b', label='portfolio')
    ax1_2.plot(results.index, results['buy_hold_value'], 'r--', label='buy_hold')
    ax1_2.set_ylabel('value', fontsize=12)
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    # 标记交易点
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        # 买入点
        buy_trades = trades_df[trades_df['type'] == '买入']
        if len(buy_trades) > 0:
            ax1.scatter(buy_trades['date'], buy_trades['price'],
                        marker='^', color='g', s=100, label='买入')
        # 卖出点
        sell_trades = trades_df[trades_df['type'] == '卖出']
        if len(sell_trades) > 0:
            ax1.scatter(sell_trades['date'], sell_trades['price'],
                        marker='v', color='r', s=100, label='卖出')
        # 止盈点
        take_profit_trades = trades_df[trades_df['type'] == '止盈']
        if len(take_profit_trades) > 0:
            ax1.scatter(take_profit_trades['date'], take_profit_trades['price'],
                        marker='v', color='purple', s=100, label='止盈')
        # 止损点
        stop_loss_trades = trades_df[trades_df['type'] == '止损']
        if len(stop_loss_trades) > 0:
            ax1.scatter(stop_loss_trades['date'], stop_loss_trades['price'],
                        marker='v', color='orange', s=100, label='止损')

    # 2. 持仓图
    ax2 = axes[1]
    ax2.set_title('status', fontsize=14)
    ax2.fill_between(results.index, 0, results['position'], color='skyblue', alpha=0.5)
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['flat', 'holding'])

    # 3. 累积收益对比图
    ax3 = axes[2]
    ax3.set_title('cum_return', fontsize=14)
    ax3.plot(results.index, results['cumulative_return'] * 100, 'b', label='portfolio')
    ax3.plot(results.index, results['buy_hold_cumulative_return'] * 100, 'r--', label='buy_hold')
    ax3.set_ylabel('return %', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 添加绩效指标文本框
    textstr = '\n'.join((
        f"策略表现:",
        f"累计回报: {metrics['累计回报']:.2%}",
        f"年化收益率: {metrics['年化收益率']:.2%}",
        f"夏普比率: {metrics['夏普比率']:.2f}",
        f"最大回撤: {metrics['最大回撤']:.2%}",
        f"胜率: {metrics['胜率']:.2%}",
        f"交易次数: {metrics['交易次数']}",
        f"\n买入持有表现:",
        f"累计回报: {metrics['买入持有累计回报']:.2%}",
        f"年化收益率: {metrics['买入持有年化收益率']:.2%}",
        f"夏普比率: {metrics['买入持有夏普比率']:.2f}",
        f"最大回撤: {metrics['买入持有最大回撤']:.2%}"
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontproperties=my_font)
    plt.tight_layout()
    return fig

# 主函数
def main():
    print("加载数据...")
    btc_prices = pd.read_hdf('data/btc_prices.h5')
    btc_prices['daily_return'] = btc_prices['price'].pct_change(1)  # T日放置T-1到T的一日收益率
    btc_prices['daily_return'] = btc_prices['daily_return'].shift(-1)  # T日放置T到T+1的一日收益率
    btc_prices = btc_prices.dropna()
    etf_data = pd.read_csv('data/etf_net_flow.csv', parse_dates=['date'], index_col='date')
    etf_data = etf_data.replace('-', np.nan).astype(float)
    btc_prices.index = pd.to_datetime(btc_prices.index)
    etf_data.index = pd.to_datetime(etf_data.index)
    etf_data = etf_data['IBIT']
    # T日只能获取T-1的ETF信息
    etf_data.index = pd.Series(etf_data.index).apply(lambda x: x + pd.Timedelta('1D'))
    etf_data = etf_data.dropna()

    # 生成信号
    merged_data = pd.merge(btc_prices, etf_data, left_index=True, right_index=True, how='inner')
    merged_data['signal'] = 0
    merged_data.loc[merged_data['IBIT'] < 0, 'signal'] = 1  # 做多信号: IBIT资金净流出
    merged_data.loc[merged_data['IBIT'] > 0, 'signal'] = -1  # 平仓/做空信号: IBIT资金净流入

    print("执行回测...")
    results, trades = run_backtest(merged_data, initial_capital=100000, stop_loss_pct=-0.10, take_profit_pct=0.20)

    print("计算指标...")
    metrics = calculate_metrics(results, trades, merged_data['daily_return'])

    # 打印性能指标
    print("\n回测性能指标:")
    for key, value in metrics.items():
        if '率' in key or '回报' in key or '回撤' in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")

    # 可视化结果
    fig = plot_results(results, trades, metrics)
    plt.figure(fig.number)
    plt.savefig('backtest_results.png')
    plt.show()
    return

# 如果直接运行此脚本，则执行main函数
if __name__ == "__main__":
    main()
