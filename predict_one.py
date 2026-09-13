# predict_one.py (fixed)
import os, argparse, math, numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from features_loader import load_symbol_parquet

def main(root, ticker, start=None, val_years=2, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 读取 + 特征
    df = load_symbol_parquet(root, ticker, start=start)

    # 2) 目标：明日对数收益
    y = (np.log(df["close"]).shift(-1) - np.log(df["close"])).dropna()
    X = df.loc[y.index]

    # —— 仅保留数值列，清理无效值 ——（关键修复）
    X = X.select_dtypes(include=["number", "bool"]).copy()
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype("int8")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    # 3) 时间切分（最近 val_years 年做验证）
    split_date = X.index.max() - pd.DateOffset(years=val_years)
    X_tr, y_tr = X[X.index<=split_date], y[y.index<=split_date]
    X_va, y_va = X[X.index> split_date], y[y.index> split_date]
    if len(X_tr) < 200 or len(X_va) < 20:
        raise RuntimeError("样本太少，换一只股票或缩小 val_years。")

    # 4) 训练
    model = LGBMRegressor(
        n_estimators=800, learning_rate=0.03,
        num_leaves=63, subsample=0.9, colsample_bytree=0.9,
        random_state=2025
    )
    model.fit(X_tr, y_tr)

    # 5) 验证评估
    pred_va = pd.Series(model.predict(X_va), index=X_va.index, name="pred")
    rmse = float(np.sqrt(np.mean((pred_va - y_va)**2)))
    direction_acc = float((np.sign(pred_va) == np.sign(y_va)).mean())

    # 6) 明日预测（基于最后一天特征）
    x_last = X.iloc[[-1]]
    y_next = float(model.predict(x_last)[0])     # 预测 next-day log-return
    last_close = float(df["close"].iloc[-1])
    pred_next_close = math.exp(y_next) * last_close

    # 7) 输出与落盘
    print(f"[{ticker}] val_years={val_years} RMSE={rmse:.6f} 方向准确率={direction_acc:.3f}")
    print(f"最后收盘价={last_close:.4f}  预测明日log-ret={y_next:.5f}  预测明日收盘≈{pred_next_close:.4f}")

    out_csv = os.path.join(out_dir, f"{ticker}_valid_pred.csv")
    pd.DataFrame({"y": y_va, "pred": pred_va}).to_csv(out_csv, encoding="utf-8-sig")
    print("验证曲线已保存 ->", out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="parquet_by_symbol 目录")
    ap.add_argument("--ticker", required=True, help="如 sz000671")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--val_years", type=int, default=2)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    main(args.root, args.ticker, args.start, args.val_years, args.out_dir)
