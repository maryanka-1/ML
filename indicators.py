import numpy as np
import pandas as pd

# Relative strength index, индекс относительной силы(моменты, когда цена актива выросла или упала слишком сильно)
def RSI(prices, n_steps=14):
    prices = np.array(prices)
    rsi_values = [None] * n_steps
    for i in range(n_steps, len(prices)):
        current_prices = prices[i - n_steps + 1 : i + 1]
        ups = []
        downs = []
        price_changes = current_prices[1:] - current_prices[:-1]
        for price_change in price_changes:
            if price_change >= 0:
                ups.append(price_change)
                downs.append(0)
            else:
                ups.append(0)
                downs.append(-price_change)
        avg_up = np.mean(ups)
        avg_down = np.mean(downs)

        rsi_val = 100 - 100 / (1 + avg_up / avg_down)
        rsi_values.append(rsi_val)

    return rsi_values


# Bollinger Bands(Полосы Боллинджера, текущие отклонения цены)
def extract_bb(prices, n_steps: int = 14):
    bollinger_list = [None] * n_steps

    for i in range(n_steps, len(prices)):
        mean_price = prices.iloc[i - n_steps : i + 1].mean()
        sigma = prices.iloc[i - n_steps : i + 1].std()
        current_price = prices.iloc[i]

        result = (current_price - mean_price) / sigma
        bollinger_list.append(result)
    return bollinger_list


# Money Flow Index(для измерения давления покупки и продажи)
def MFI(open, high, low, close, volume, n_steps: int = 14):
    mfi_list = [None] * n_steps

    for i in range(n_steps, len(close)):
        open_i = open.iloc[i - n_steps : i + 1]
        high_i = high.iloc[i - n_steps : i + 1]
        low_i = low.iloc[i - n_steps : i + 1]
        close_i = close.iloc[i - n_steps : i + 1]
        volume_i = volume.iloc[i - n_steps : i + 1]
        typical_price = (high_i + low_i + close_i) / 3
        raw_money_flow = typical_price * volume_i

        mask = (close_i - open_i) > 0
        positive_flow = (raw_money_flow[mask]).sum()
        negative_flow = (raw_money_flow[~mask]).sum()

        ratio = positive_flow / negative_flow
        mfi_val = 100 - 100 / (1 + ratio)

        mfi_list.append(mfi_val)
    return mfi_list


# "Chaikin Money Flow", денежный поток Чайкина(изменение давления покупок и продаж, на основе объёмов и изменение цены.)
def CMF(open, high, low, close, volume, n_steps: int = 14):
    cmf_list = [None] * n_steps

    for i in range(n_steps, len(close)):
        open_i = open.iloc[i - n_steps : i + 1]
        high_i = high.iloc[i - n_steps : i + 1]
        low_i = low.iloc[i - n_steps : i + 1]
        close_i = close.iloc[i - n_steps : i + 1]
        volume_i = volume.iloc[i - n_steps : i + 1]

        multiplier = ((close_i - low_i) - (high_i - close_i)) / (high_i - low_i)
        mf_vol = multiplier * volume_i

        cmf_val = mf_vol.sum() / volume_i.sum()

        cmf_list.append(cmf_val)
    return cmf_list

# Average True Range (средний истинный диапазон).
def ATR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n_steps: int = 14,
):
    atr_list = [None] * n_steps

    for i in range(n_steps, len(high)):
        first = int(i == n_steps)
        high_i = high.iloc[i - n_steps + first : i + 1].values
        low_i = low.iloc[i - n_steps + first : i + 1].values
        close_i = close.iloc[i - n_steps + first : i + 1].values
        atr = ((high_i - low_i) / close_i * 100).mean()
        atr_list.append(atr)

    return atr_list


# индикатор среднего направленного движения ADX (от англ. Average Directional Movement Index)
def ADX(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n_steps: int = 14,
    alpha: float = 0.2,
):
    adx_list = [None] * n_steps

    for i in range(n_steps, len(close)):
        first = int(i == n_steps)
        high_i = high.iloc[i - n_steps + first : i + 1].values
        low_i = low.iloc[i - n_steps + first : i + 1].values
        close_i = close.iloc[i - n_steps + first : i + 1].values

        TR = np.maximum(high_i[1:] - low_i[1:], high_i[1:] - close_i[:-1])

        pos_M, neg_M = (
            high_i[1:] - high_i[:-1],
            low_i[:-1] - low_i[1:],
        )
        pos_M[np.logical_or(pos_M < neg_M, pos_M < 0)] = 0
        neg_M[np.logical_or(pos_M > neg_M, pos_M < 0)] = 0

        pos_DI = pd.Series(pos_M / TR).ewm(alpha=alpha).mean()
        neg_DI = pd.Series(neg_M / TR).ewm(alpha=alpha).mean()

        adx_val = (
            100
            * ((pos_DI - neg_DI) / (pos_DI + neg_DI)).ewm(alpha=alpha).mean().iloc[-1]
        )
        adx_list.append(adx_val)

    return adx_list
