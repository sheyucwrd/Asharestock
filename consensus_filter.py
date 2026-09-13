import pandas as pd
import argparse, os

def load_df(path, name):
    f = os.path.join(path, "today_signals.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{name} 没有找到 today_signals.csv")
    df = pd.read_csv(f)
    print(f"[{name}] 加载 {len(df)} 行")
    return df

def merge_consensus(lstm_path, lgbm_path, acc_min=0.52, k=1.0, out_dir="consensus_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df_lstm = load_df(lstm_path, "LSTM")
    df_lgbm = load_df(lgbm_path, "LGBM")

    # 只保留关键列
    cols = ["ticker","y_next_pred","direction_acc_valid","z_of_pred"]
    for df in [df_lstm, df_lgbm]:
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"{c} 不在 {df.columns.tolist()} 中")

    df_lstm = df_lstm[cols].rename(columns={
        "y_next_pred":"y_pred_lstm",
        "direction_acc_valid":"acc_lstm",
        "z_of_pred":"z_lstm"
    })
    df_lgbm = df_lgbm[cols].rename(columns={
        "y_next_pred":"y_pred_lgbm",
        "direction_acc_valid":"acc_lgbm",
        "z_of_pred":"z_lgbm"
    })

    df = pd.merge(df_lstm, df_lgbm, on="ticker", suffixes=("_lstm","_lgbm"))
    print("合并后股票数：", len(df))

    # 同方向一致：涨 or 跌
    same_dir = ((df["y_pred_lstm"] > 0) & (df["y_pred_lgbm"] > 0)) | \
               ((df["y_pred_lstm"] < 0) & (df["y_pred_lgbm"] < 0))

    # 置信条件：任一模型准确率≥acc_min 且 |z|≥k
    conf = (
        ((abs(df["z_lstm"]) >= k) & (df["acc_lstm"] >= acc_min)) |
        ((abs(df["z_lgbm"]) >= k) & (df["acc_lgbm"] >= acc_min))
    )

    df_cons = df[same_dir & conf].copy()
    df_cons["direction"] = df_cons.apply(lambda r: "Up" if r["y_pred_lstm"] > 0 else "Down", axis=1)
    df_cons["z_mean"] = df_cons[["z_lstm","z_lgbm"]].abs().mean(axis=1)
    df_cons = df_cons.sort_values("z_mean", ascending=False)

    up = df_cons[df_cons["direction"]=="Up"]
    down = df_cons[df_cons["direction"]=="Down"]

    out_up = os.path.join(out_dir, "consensus_up.csv")
    out_down = os.path.join(out_dir, "consensus_down.csv")
    out_all = os.path.join(out_dir, "consensus_all.csv")

    up.to_csv(out_up, index=False, encoding="utf-8-sig")
    down.to_csv(out_down, index=False, encoding="utf-8-sig")
    df_cons.to_csv(out_all, index=False, encoding="utf-8-sig")

    print(f"输出：{len(df_cons)} 条共识信号")
    print(f" - 上涨：{len(up)} 条  → {out_up}")
    print(f" - 下跌：{len(down)} 条 → {out_down}")
    print(f"全部信号 → {out_all}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", required=True, help="LSTM 输出目录路径（含 today_signals.csv）")
    ap.add_argument("--lgbm", required=True, help="LGBM 输出目录路径（含 today_signals.csv）")
    ap.add_argument("--acc_min", type=float, default=0.52)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--out_dir", default="consensus_outputs")
    args = ap.parse_args()

    merge_consensus(args.lstm, args.lgbm, args.acc_min, args.k, args.out_dir)
