
import sys, os, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "reduced_memory_all_A_stock_data.pkl"
assert os.path.exists(path), f"File not found: {path}"

print(f"Loading: {path}")
obj = pd.read_pickle(path)

print("\n=== Basic Info ===")
print("Type:", type(obj))
if isinstance(obj, pd.DataFrame):
    df = obj
else:
    # 有些 pkl 里是 dict/list，尝试取第一个 DataFrame
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, pd.DataFrame):
                df = v; print(f"Found DataFrame in dict key: {k}"); break
        else:
            raise TypeError("No DataFrame found inside dict pkl.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, pd.DataFrame):
                df = v; print(f"Found DataFrame in list index: {i}"); break
        else:
            raise TypeError("No DataFrame found inside list pkl.")
    else:
        raise TypeError("Unsupported pkl content type.")

print(df.head(3))
print("\nShape:", df.shape)
mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
print(f"Memory (deep): {mem_mb:.2f} MB")

print("\n=== Dtypes ===")
print(df.dtypes)

# 尝试识别常见列名
cands_symbol = [c for c in df.columns if c.lower() in ["symbol","ts_code","code","ticker","sec_code","stock_code"]]
cands_date   = [c for c in df.columns if c.lower() in ["date","trade_date","datetime","time","dt"]]
print("\nGuess -> symbol:", cands_symbol, " date:", cands_date)

# 规范：转日期，设置 MultiIndex（symbol,date）
sym = cands_symbol[0] if cands_symbol else None
dt  = cands_date[0]   if cands_date   else None

if sym and dt:
    df = df.copy()
    df[dt] = pd.to_datetime(df[dt], errors="coerce")
    df = df.dropna(subset=[dt])
    df = df.sort_values([sym, dt])
    df = df.set_index([sym, dt])
    print("\nIndex set to:", df.index.names)
else:
    print("\n未自动识别到 symbol/date 列，后面步骤会跳过索引标准化。")

# 保存一个小切片，方便快速检查
out_sample = "sample_5_stocks.csv"
if isinstance(df.index, pd.MultiIndex) and df.index.names == [sym, dt]:
    small = df.groupby(level=0, sort=False).head(200)  # 每只取200行
else:
    small = df.head(1000)
small.to_csv(out_sample, index=True)
print(f"\nWrote preview: {out_sample}")
