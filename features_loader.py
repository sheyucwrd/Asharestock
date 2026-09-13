# features_loader.py
import os, pandas as pd, numpy as np

def load_symbol_parquet(root_dir, ticker,
                        start=None, end=None,
                        price_col="close", vol_col="volume",
                        add_feats=True):
    """
    从按股票分区目录读取一只股票（可加时间窗），并生成基础时序特征：
    - log returns: ret1, ret5
    - 移动均值/波动率: ma5, ma20, vol5
    - 成交量对数: log_vol
    """
    path = os.path.join(root_dir, f"{ticker}.parquet")
    if not os.path.exists(path):
        # 兼容元数据里替换过符号的情况
        path = os.path.join(root_dir, f"{ticker}".replace("/", "_").replace("\\", "_").replace(":", "_") + ".parquet")
    df = pd.read_parquet(path)

    # 找到日期列并设为索引
    date_col = next((c for c in ["date","trade_date","datetime","time","dt"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("日期列未找到")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # 时间窗
    if start: df = df[df.index >= pd.to_datetime(start)]
    if end:   df = df[df.index <= pd.to_datetime(end)]

    if add_feats and price_col in df:
        px = df[price_col].astype("float32")
        df["ret1"] = np.log(px).diff(1)
        df["ret5"] = np.log(px).diff(5)
        df["ma5"]  = px.rolling(5).mean()
        df["ma20"] = px.rolling(20).mean()
        df["vol5"] = px.pct_change().rolling(5).std()
    if add_feats and vol_col in df:
        df["log_vol"] = np.log1p(df[vol_col].astype("float32"))

    df = df.dropna()
    return df

if __name__ == "__main__":
    root = r"C:\Users\Administrator\Desktop\AshareSTockPrice\parquet_by_symbol"
    # 示例：读取某只股票的 2018-2024
    try:
        sample = load_symbol_parquet(root, "sz000671", start="2018-01-01", end="2024-12-31")
        print(sample.head())
        print(sample.tail())
        print("shape:", sample.shape)
    except Exception as e:
        print("Error:", e)
