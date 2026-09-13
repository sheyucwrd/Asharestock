import pandas as pd
import argparse, os

def filter_strong_signals(path, acc_min=0.55, k=1.0, out_dir="strong_outputs"):
    f = os.path.join(path, "today_signals.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"找不到 {f}")
    
    df = pd.read_csv(f)
    print(f"加载 {len(df)} 条信号")

    # 必要列检查
    for c in ["ticker", "y_next_pred", "direction_acc_valid", "z_of_pred"]:
        if c not in df.columns:
            raise ValueError(f"列 {c} 缺失, 当前列: {df.columns.tolist()}")

    # 强信号过滤条件
    df_strong = df[
        (abs(df["z_of_pred"]) >= k) & 
        (df["direction_acc_valid"] >= acc_min)
    ].copy()

    df_strong["direction"] = df_strong["y_next_pred"].apply(lambda x: "Up" if x > 0 else "Down")
    df_strong = df_strong.sort_values("z_of_pred", key=abs, ascending=False)

    os.makedirs(out_dir, exist_ok=True)

    up = df_strong[df_strong["direction"] == "Up"]
    down = df_strong[df_strong["direction"] == "Down"]

    up.to_csv(os.path.join(out_dir, "strong_up.csv"), index=False, encoding="utf-8-sig")
    down.to_csv(os.path.join(out_dir, "strong_down.csv"), index=False, encoding="utf-8-sig")
    df_strong.to_csv(os.path.join(out_dir, "strong_all.csv"), index=False, encoding="utf-8-sig")

    print(f"共输出 {len(df_strong)} 条强信号：上涨 {len(up)}，下跌 {len(down)}")
    print(f"输出目录：{out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="LSTM 输出目录路径（含 today_signals.csv）")
    ap.add_argument("--acc_min", type=float, default=0.55)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--out_dir", default="strong_outputs")
    args = ap.parse_args()

    filter_strong_signals(args.path, args.acc_min, args.k, args.out_dir)
