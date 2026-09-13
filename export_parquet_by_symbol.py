# export_parquet_by_symbol.py  (robust)
import os, sys, glob, pandas as pd, numpy as np

def pick(cols, names):
    for n in names:
        for c in cols:
            if c.lower() == n:
                return c
    return None

# ---- 解析参数：src pkl, out_dir ----
if len(sys.argv) >= 2:
    src = sys.argv[1]
else:
    cand = glob.glob("*.pkl")
    assert cand, "当前目录找不到 .pkl，请传入 pkl 路径：python export_parquet_by_symbol.py <src.pkl> [out_dir]"
    assert len(cand) == 1, f"当前目录存在多个 pkl：{cand}，请显式指定要转换的文件。"
    src = cand[0]

out_dir = sys.argv[2] if len(sys.argv) >= 3 else "parquet_by_symbol"
os.makedirs(out_dir, exist_ok=True)

print("Loading pkl ...", os.path.abspath(src))
df = pd.read_pickle(src)

# 删除无用列
for col in ["Unnamed: 0"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# 识别列名
TICKER = pick(df.columns, ["ticker","ts_code","code","ticker","sec_code","stock_code"]) or "ticker"
DATE   = pick(df.columns, ["date","trade_date","datetime","time","dt"]) or "date"
if TICKER not in df.columns or DATE not in df.columns:
    raise ValueError(f"未找到必要列：ticker/date，现有列：{list(df.columns)[:10]} ...")

# 规范 dtype/排序
df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
df = df.dropna(subset=[DATE]).sort_values([TICKER, DATE])

# 降精度
float_cols = df.select_dtypes(include=["float64","float32"]).columns.tolist()
for c in float_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

print("Writing per-symbol parquet to:", os.path.abspath(out_dir))
meta = []
for sid, g in df.groupby(TICKER, sort=False):
    fn = f"{sid}".replace("/", "_").replace("\\", "_").replace(":", "_")
    p = os.path.join(out_dir, f"{fn}.parquet")
    g.to_parquet(p, engine="pyarrow", compression="snappy", index=False)
    meta.append({
        "ticker": sid,
        "rows": len(g),
        "start_date": g[DATE].iloc[0].date(),
        "end_date": g[DATE].iloc[-1].date(),
        "file": p
    })

meta_df = pd.DataFrame(meta).sort_values("ticker")
meta_csv = os.path.join(out_dir, "_metadata.csv")
meta_df.to_csv(meta_csv, index=False, encoding="utf-8-sig")
print("Done. Symbols:", len(meta_df), "Total rows:", int(meta_df['rows'].sum()))
print("Metadata ->", meta_csv)
