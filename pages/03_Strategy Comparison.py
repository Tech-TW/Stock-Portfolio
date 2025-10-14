# ──────────────────────────────────────────────────────────
# 檔案：pages/03_📊_策略比較.py
# 說明：比較標的條件設定 + 鏡像、DCA、Lump Sum 比較分析與曲線
# 依賴：Upload 頁面已把 df 放入 st.session_state["uploaded_df"]
# ──────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from collections import deque
from dateutil.relativedelta import relativedelta
import yfinance as yf

st.title("📊 策略比較（鏡像 / DCA / Lump Sum）")

# 取得上傳資料
if "uploaded_df" not in st.session_state or st.session_state["uploaded_df"] is None:
    st.error("No data. Please upload at the **Upload** page first.")
    st.stop()

df_input: pd.DataFrame = st.session_state["uploaded_df"].copy()

# ========= 共用函式（與 02 頁一致；為了單檔可用，這裡再貼一次）=========
def make_excel_report(dfs: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, _df in dfs.items():
            safe = str(name)[:31]
            (_df if isinstance(_df, pd.DataFrame) else pd.DataFrame(_df)).to_excel(
                writer, sheet_name=safe, index=False
            )
    buf.seek(0)
    return buf.read()

def determine_currency(ticker: str) -> str:
    t = str(ticker).upper()
    if t.endswith(".TW") or t.endswith(".TWO"): return "TWD"
    if t.isdigit() and len(t) == 4:            return "TWD"
    if t.endswith(".HK"):                       return "HKD"
    if t.endswith(".T"):                        return "JPY"
    if t.endswith(".L"):                        return "GBP"
    if t.endswith(".DE") or t.endswith(".F"):   return "EUR"
    if t.endswith(".TO"):                       return "CAD"
    if t.endswith(".AX"):                       return "AUD"
    return "USD"

def download_fx_history(currency: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if currency == "TWD":
        return pd.DataFrame({"日期":[start_date], "匯率":[1.0], "幣別":["TWD"]})
    try:
        fx_symbol = f"{currency}TWD=X"
        start_ex = start_date - timedelta(days=10)
        end_ex   = end_date + timedelta(days=10)
        fx = yf.download(fx_symbol, start=start_ex, end=end_ex, auto_adjust=True, progress=False)
        if fx.empty:
            return pd.DataFrame()
        fx_df = fx["Close"].reset_index()
        fx_df.columns = ["日期","匯率"]
        fx_df["日期"]  = pd.to_datetime(fx_df["日期"]).dt.normalize()
        fx_df["幣別"]  = currency
        return fx_df
    except Exception:
        return pd.DataFrame()

def get_fx_rate(date: pd.Timestamp, currency: str, fx_data_dict: dict) -> float:
    if currency == "TWD":
        return 1.0
    f = fx_data_dict.get(currency)
    if f is None or f.empty:
        return np.nan
    d0 = f.loc[(f["日期"] <= date), "日期"].max()
    if pd.isna(d0):
        d0 = f["日期"].min()
    rate = f.loc[f["日期"]==d0, "匯率"]
    return float(rate.iloc[0]) if not rate.empty else np.nan

def get_latest_fx_rate(currency: str, fx_data_dict: dict) -> float:
    if currency == "TWD":
        return 1.0
    f = fx_data_dict.get(currency)
    if f is None or f.empty:
        try:
            sym = f"{currency}TWD=X"
            fx2 = yf.download(sym, period="5d", interval="1d", auto_adjust=True, progress=False)
            if not fx2.empty:
                return float(fx2["Close"].dropna().iloc[-1])
        except Exception:
            pass
        return np.nan
    f2 = f.dropna(subset=["匯率"]).sort_values("日期", ascending=False)
    if f2.empty: return np.nan
    return float(f2.iloc[0]["匯率"])

def download_stock_history(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        start_ex = start_date - timedelta(days=10)
        end_ex   = end_date + timedelta(days=10)
        s = yf.download(ticker, start=start_ex, end=end_ex, auto_adjust=True, progress=False)
        if s.empty:
            return pd.DataFrame()
        sdf = s["Close"].reset_index()
        sdf.columns = ["日期","收盤價"]
        sdf["日期"] = pd.to_datetime(sdf["日期"]).dt.normalize()
        sdf["股票代號"] = ticker
        return sdf
    except Exception:
        return pd.DataFrame()

# ========= 股票分割事件處理（如需用到平均成本基礎曲線時可擴充） =========
class StockEventProcessor:
    def __init__(self):
        self.events_data = {}

    def fetch_stock_events(self, ticker, start_date, end_date):
        events = {'splits': []}
        try:
            stock = yf.Ticker(ticker)
            actions = stock.actions.reset_index()
            if actions.empty:
                self.events_data[ticker] = events
                return events
            actions["Date"] = actions["Date"].dt.tz_localize(None)
            actions = actions[(actions["Date"]>=start_date)&(actions["Date"]<=end_date)].copy()
            if "Stock Splits" in actions.columns:
                s = actions[actions["Stock Splits"]>0][["Date","Stock Splits"]]
                if not s.empty:
                    s.columns = ["Date","Split_Ratio"]
                    for _,r in s.iterrows():
                        events["splits"].append({"date": r["Date"], "ratio": r["Stock Splits"], "type": "split"})
        except Exception:
            pass
        self.events_data[ticker] = events
        return events

# ===== 投資比較：輔助函式 =====
def get_price_on_or_before(date, ticker, stock_data_dict, min_date, max_date):
    if ticker not in stock_data_dict or stock_data_dict[ticker].empty:
        _df = download_stock_history(ticker, min_date, max_date)
        stock_data_dict[ticker] = _df if not _df.empty else pd.DataFrame(columns=["日期","收盤價","股票代號"])
    sdf = stock_data_dict[ticker]
    if sdf.empty: return np.nan
    sdf = sdf.sort_values("日期")
    mask = sdf["日期"] <= pd.to_datetime(date).normalize()
    if not mask.any():
        return float(sdf["收盤價"].iloc[0])
    return float(sdf.loc[mask, "收盤價"].iloc[-1])

def make_mirror_trades(df_trades, target_ticker, fx_data_dict, stock_data_dict, min_date, max_date):
    rows = []
    for _, r in df_trades.sort_values("日期").iterrows():
        d = pd.to_datetime(r["日期"]).normalize()
        shares = float(r["購買股數"])
        price  = float(r["購買股價"])
        fx_used = float(r.get("換匯匯率", 1.0))
        tx_cost_foreign = float(r.get("交易成本", 0.0))

        gross_foreign = price * abs(shares)
        tx_cost_twd   = tx_cost_foreign * fx_used
        cash_twd_abs  = gross_foreign * fx_used + tx_cost_twd
        sign = 1 if shares > 0 else -1

        px_t = get_price_on_or_before(d, target_ticker, stock_data_dict, min_date, max_date)
        ccy_t = determine_currency(target_ticker)
        fx_t  = get_fx_rate(d, ccy_t, fx_data_dict)
        if np.isnan(px_t) or np.isnan(fx_t) or px_t <= 0 or fx_t <= 0:
            continue

        tx_cost_foreign_target = tx_cost_twd / fx_t
        notional_twd_abs = gross_foreign * fx_used
        shares_target = (notional_twd_abs / (px_t * fx_t)) * sign

        rows.append({
            "日期": d, "股票代號": target_ticker, "購買股數": shares_target,
            "購買股價": px_t, "換匯匯率": fx_t, "交易成本": tx_cost_foreign_target, "幣別": ccy_t
        })
    df_alt = pd.DataFrame(rows)
    for col in ["投資金額","交易金額"]:
        if col not in df_alt.columns: df_alt[col] = np.nan
    return df_alt

def make_monthly_dca_trades(start_date, end_date, amount_twd, target_ticker, fx_data_dict, stock_data_dict, dca_day=1):
    rows = []
    cur = pd.to_datetime(start_date).replace(day=dca_day)
    end_date = pd.to_datetime(end_date)
    while cur <= end_date:
        px = get_price_on_or_before(cur, target_ticker, stock_data_dict, start_date, end_date)
        ccy = determine_currency(target_ticker)
        fx  = get_fx_rate(cur, ccy, fx_data_dict)
        if (not np.isnan(px)) and (not np.isnan(fx)) and px > 0 and fx > 0:
            shares = (amount_twd / (px * fx))
            rows.append({
                "日期": cur.normalize(), "股票代號": target_ticker, "購買股數": shares,
                "購買股價": px, "換匯匯率": fx, "交易成本": 0.0, "幣別": ccy
            })
        cur = cur + relativedelta(months=1)
    df_dca = pd.DataFrame(rows)
    for col in ["投資金額","交易金額"]:
        if col not in df_dca.columns: df_dca[col] = np.nan
    return df_dca

def make_lumpsum_trades_from_budget(df_all: pd.DataFrame, target_ticker: str,
                                    fx_data_dict: dict, stock_data_dict: dict,
                                    start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if "投資預算總水位" not in df_all.columns or "日期" not in df_all.columns:
        return pd.DataFrame(columns=["日期","股票代號","購買股數","購買股價","換匯匯率","交易成本","幣別"])

    ser = (df_all[["日期","投資預算總水位"]]
           .dropna(subset=["日期"])
           .assign(日期=lambda x: pd.to_datetime(x["日期"]).dt.normalize())
           .sort_values(["日期"])
           .drop_duplicates(subset=["日期"], keep="last"))
    ser["投資預算總水位"] = pd.to_numeric(ser["投資預算總水位"], errors="coerce")
    ser = ser.dropna(subset=["投資預算總水位"])
    ser = ser[(ser["日期"] >= pd.to_datetime(start_date)) & (ser["日期"] <= pd.to_datetime(end_date))]

    if ser.empty:
        return pd.DataFrame(columns=["日期","股票代號","購買股數","購買股價","換匯匯率","交易成本","幣別"])

    ser["prev"]  = ser["投資預算總水位"].shift(1).fillna(0.0)
    ser["delta"] = ser["投資預算總水位"] - ser["prev"]
    ser = ser[ser["delta"] > 0]

    rows = []
    tgt_ccy = determine_currency(target_ticker)
    for _, r in ser.iterrows():
        d = r["日期"]
        amt_twd = float(r["delta"])
        px = get_price_on_or_before(d, target_ticker, stock_data_dict, start_date, end_date)
        fx = get_fx_rate(d, tgt_ccy, fx_data_dict)
        if np.isnan(px) or np.isnan(fx) or px <= 0 or fx <= 0 or amt_twd <= 0:
            continue
        shares = amt_twd / (px * fx)
        rows.append({
            "日期": d, "股票代號": target_ticker, "購買股數": shares,
            "購買股價": px, "換匯匯率": fx, "交易成本": 0.0, "幣別": tgt_ccy
        })
    df_ls = pd.DataFrame(rows)
    for col in ["投資金額","交易金額"]:
        if col not in df_ls.columns: df_ls[col] = np.nan
    return df_ls

# === 權益曲線（針對任意「交易明細 df」） ===
def equity_curve_for_trades(df_trades_like: pd.DataFrame, all_dates, stock_close_daily, fx_daily, latest_prices, base_index) -> pd.DataFrame:
    if df_trades_like is None or df_trades_like.empty:
        return pd.DataFrame(columns=["日期", "權益(台幣)"])

    pos = {}
    cum_realized_twd = 0.0
    rows = []

    df_like = df_trades_like.copy()
    df_like["日期"] = pd.to_datetime(df_like["日期"]).dt.normalize()
    df_like["交易成本"] = pd.to_numeric(df_like.get("交易成本", 0.0), errors="coerce").fillna(0.0)
    df_like["換匯匯率"] = pd.to_numeric(df_like.get("換匯匯率", 1.0), errors="coerce").fillna(1.0)
    df_like["幣別"] = df_like.get("幣別", df_like["股票代號"].apply(determine_currency))

    like_by_day = {d: g for d, g in df_like.sort_values("日期").groupby("日期")}
    last_day_local = all_dates[-1]

    for day in all_dates:
        if day in like_by_day:
            for _, r in like_by_day[day].iterrows():
                tkr = r["股票代號"]; sh = float(r["購買股數"]); px = float(r["購買股價"])
                fx  = float(r.get("換匯匯率", 1.0)); fee = float(r.get("交易成本", 0.0))
                ccy = r.get("幣別", determine_currency(tkr))

                if tkr not in pos:
                    pos[tkr] = {"shares":0.0, "avg_cost_foreign":0.0, "avg_fx":1.0,
                                "total_cost_twd":0.0, "currency":ccy}

                p = pos[tkr]
                if sh > 0:
                    actual = (px * sh + fee) / sh
                    new_sh = p["shares"] + sh
                    new_cf = p["avg_cost_foreign"] * p["shares"] + actual * sh
                    new_ct = p["total_cost_twd"] + actual * sh * fx
                    p["shares"] = new_sh
                    if new_sh > 0:
                        p["avg_cost_foreign"] = new_cf / new_sh
                        p["avg_fx"]           = (new_ct / new_cf) if new_cf > 0 else 1.0
                        p["total_cost_twd"]   = p["avg_cost_foreign"] * p["shares"] * p["avg_fx"]
                    else:
                        p["avg_cost_foreign"] = 0.0
                        p["avg_fx"] = 1.0
                        p["total_cost_twd"] = 0.0
                else:
                    sell = abs(sh)
                    if p["shares"] <= 0 or p["shares"] < sell:
                        continue
                    gross = px * sell
                    net   = gross - fee
                    real_f = (net / sell - p["avg_cost_foreign"]) * sell
                    real_t = real_f * p["avg_fx"]
                    cum_realized_twd += real_t
                    p["shares"] -= sell
                    if p["shares"] > 0:
                        p["total_cost_twd"] = p["avg_cost_foreign"] * p["shares"] * p["avg_fx"]
                    else:
                        p["avg_cost_foreign"] = 0.0
                        p["avg_fx"] = 1.0
                        p["total_cost_twd"] = 0.0

        total_mv_twd = 0.0
        for tkr, p in pos.items():
            if p["shares"] <= 0:
                continue
            ccy = p["currency"]
            if day == last_day_local:
                px_today = latest_prices.get(tkr, np.nan)
                if np.isnan(px_today):
                    series = stock_close_daily.get(tkr)
                    px_today = float(series.get(day, np.nan)) if series is not None else np.nan
                fx_today = float(fx_daily.get(ccy, pd.Series(index=all_dates)).iloc[-1]) if ccy in fx_daily else (1.0 if ccy=="TWD" else np.nan)
            else:
                series = stock_close_daily.get(tkr)
                px_today = float(series.get(day, np.nan)) if series is not None else np.nan
                fx_today = float(fx_daily.get(ccy, pd.Series(index=all_dates)).get(day, np.nan))
            if np.isnan(px_today) or np.isnan(fx_today):
                continue
            total_mv_twd += px_today * p["shares"] * fx_today

        total_equity_twd = total_mv_twd + cum_realized_twd
        rows.append({"日期": day, "權益(台幣)": round(total_equity_twd, 0)})

    out = pd.DataFrame(rows)
    out = out.merge(base_index, on="日期", how="right")  # 對齊全域日期
    return out

# ========= 主流程（含比較）=========
def run_full_comparison(trades_df: pd.DataFrame, dca_amount_twd: int = 70000,
                        mirror_list=None, dca_list=None, dca_day: int = 1,
                        lumpsum_list=None, valuation_to_today: bool = True) -> dict:

    # 準備欄位
    df = trades_df.copy()
    if "日期" not in df.columns:
        raise ValueError("找不到『日期』欄，請提供「日期」欄位。")
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()

    for need in ["股票代號","購買股數","購買股價"]:
        if need not in df.columns:
            raise ValueError(f"找不到必要欄位：{need}")
    if "交易成本" not in df.columns: df["交易成本"] = 0.0
    if "換匯匯率" not in df.columns: df["換匯匯率"] = 1.0
    if "幣別" not in df.columns:
        df["幣別"] = df["股票代號"].apply(determine_currency)

    df_trades = df[df["購買股數"].notna() & df["購買股價"].notna()].copy()
    df_trades["交易成本"]   = pd.to_numeric(df_trades["交易成本"], errors="coerce").fillna(0.0)
    df_trades["換匯匯率"]   = pd.to_numeric(df_trades["換匯匯率"], errors="coerce").fillna(1.0)
    df_trades["購買股數"]   = pd.to_numeric(df_trades["購買股數"], errors="coerce").fillna(0.0)
    df_trades["購買股價"]   = pd.to_numeric(df_trades["購買股價"], errors="coerce").fillna(0.0)

    min_date = df_trades["日期"].min()
    max_date = df_trades["日期"].max()

    today_tw = pd.Timestamp.today(tz="Asia/Taipei").normalize().tz_localize(None)
    end_of_range = today_tw if valuation_to_today else max_date

    # 匯率/股價歷史（原投組）
    currencies = df_trades[df_trades["幣別"]!="TWD"]["幣別"].dropna().unique()
    fx_data_dict = {}
    for cur in currencies:
        f = download_fx_history(cur, min_date, end_of_range)
        if not f.empty: fx_data_dict[cur] = f

    tickers = df_trades["股票代號"].dropna().unique()
    stock_data_dict = {}
    for tkr in tickers:
        s = download_stock_history(tkr, min_date, end_of_range)
        if not s.empty: stock_data_dict[tkr] = s

    # 構建全域日期與日頻序列容器
    all_dates = pd.date_range(min_date, end_of_range, freq="D")
    stock_close_daily = {}
    for tkr, sdf in stock_data_dict.items():
        if sdf is None or sdf.empty: continue
        ser = sdf.set_index("日期")["收盤價"].sort_index()
        ser.index = pd.to_datetime(ser.index).normalize()
        stock_close_daily[tkr] = ser.reindex(all_dates).ffill()

    fx_daily = {"TWD": pd.Series(1.0, index=all_dates)}
    for cur, fdf in fx_data_dict.items():
        if fdf is None or fdf.empty: continue
        ser = fdf.set_index("日期")["匯率"].sort_index()
        ser.index = pd.to_datetime(ser.index).normalize()
        fx_daily[cur] = ser.reindex(all_dates).ffill()

    # 比較標的與匯率預取
    default_targets = ["SPY", "0050.TW", "2330.TW"]
    mirror_targets = mirror_list if mirror_list else default_targets
    dca_targets    = dca_list if dca_list else default_targets
    lumpsum_targets= lumpsum_list if lumpsum_list else default_targets
    comp_tickers = set(mirror_targets + dca_targets + lumpsum_targets)
    comp_ccys    = {determine_currency(t) for t in comp_tickers}

    for tkr in sorted(comp_tickers):
        if tkr not in stock_data_dict or stock_data_dict[tkr] is None or stock_data_dict[tkr].empty:
            s = download_stock_history(tkr, min_date, end_of_range)
            if not s.empty:
                stock_data_dict[tkr] = s
    for tkr, sdf in stock_data_dict.items():
        if sdf is None or sdf.empty:
            continue
        if tkr not in stock_close_daily:
            ser = sdf.set_index("日期")["收盤價"].sort_index()
            ser.index = pd.to_datetime(ser.index).normalize()
            stock_close_daily[tkr] = ser.reindex(all_dates).ffill()

    for cur in sorted(c for c in comp_ccys if c != "TWD"):
        if cur not in fx_data_dict or fx_data_dict[cur] is None or fx_data_dict[cur].empty:
            f = download_fx_history(cur, min_date, end_of_range)
            if not f.empty:
                fx_data_dict[cur] = f
    for cur, fdf in fx_data_dict.items():
        if cur == "TWD" or fdf is None or fdf.empty:
            continue
        if cur not in fx_daily:
            ser = fdf.set_index("日期")["匯率"].sort_index()
            ser.index = pd.to_datetime(ser.index).normalize()
            fx_daily[cur] = ser.reindex(all_dates).ffill()
    if "TWD" not in fx_daily:
        fx_daily["TWD"] = pd.Series(1.0, index=all_dates)

    # 最新價快取
    latest_prices={}
    for tkr in comp_tickers.union(set(tickers)):
        try:
            data = yf.download(tkr, period="5d", interval="1d", auto_adjust=True, progress=False)
            if not data.empty:
                latest_prices[tkr] = float(data["Close"].dropna().iloc[-1])
        except Exception:
            pass

    # 產生比較交易集
    comparison_results = []
    compare_sheets = {}
    comparison_trade_sets = []  # (label, df_trades_like)

    # 鏡像
    for tgt in mirror_targets:
        df_m = make_mirror_trades(df_trades, tgt, fx_data_dict, stock_data_dict, min_date, end_of_range)
        if df_m.empty: continue
        disp_m = df_m.copy().sort_values("日期")
        disp_m["歷史匯率"] = disp_m["換匯匯率"]
        disp_m = disp_m[["日期","股票代號","幣別","購買股數","購買股價","換匯匯率","歷史匯率","交易成本"]]
        name = tgt.replace(".TW","").replace(".T","")
        compare_sheets[f"鏡像_{name}_買賣明細"] = disp_m
        comparison_trade_sets.append((f"鏡像-{name}", df_m))

    # DCA
    # 先暫存選項（稍後 UI 取得）
    st.session_state.setdefault("compare_defaults", ["SPY", "0050.TW", "2330.TW"])

    # 這裡的 dca_day/amount 將由 UI 控制；先給 placeholder，稍後覆寫
    # Lump Sum
    for tgt in lumpsum_targets:
        df_l = make_lumpsum_trades_from_budget(df, tgt, fx_data_dict, stock_data_dict,
                                               start_date=min_date, end_date=end_of_range)
        if df_l.empty: continue
        disp_l = df_l.copy().sort_values("日期")
        disp_l["歷史匯率"] = disp_l["換匯匯率"]
        disp_l = disp_l[["日期","股票代號","幣別","購買股數","購買股價","換匯匯率","歷史匯率","交易成本"]]
        name = tgt.replace(".TW","").replace(".T","")
        compare_sheets[f"LumpSum_{name}_買賣明細"] = disp_l
        comparison_trade_sets.append((f"LumpSum-{name}", df_l))

    # 全域日期索引（DataFrame）
    base_index = pd.DataFrame({"日期": all_dates})

    return {
        "meta": {"min_date":min_date, "max_date":max_date, "end_of_range":end_of_range},
        "fx_data_dict": fx_data_dict,
        "stock_data_dict": stock_data_dict,
        "stock_close_daily": stock_close_daily,
        "fx_daily": fx_daily,
        "latest_prices": latest_prices,
        "base_index": base_index,
        "df_trades": df_trades,
        "compare_sheets": compare_sheets,
        "comparison_trade_sets_init": comparison_trade_sets  # 鏡像/LumpSum 已放進來；DCA 由 UI 再加
    }

# ====== UI：比較標的、DCA 參數、估值日切換 ======
st.divider()
col1, col2, col3 = st.columns([2,1,1])
with col1:
    compare_choices = st.multiselect(
        "選擇比較標的（鏡像 + DCA + Lump Sum）",
        options=["SPY", "0050.TW", "2330.TW"],
        default=st.session_state.get("compare_defaults", ["SPY", "0050.TW", "2330.TW"]),
        help="三種策略都會使用這些標的。你可只選部分。"
    )
with col2:
    dca_day = st.number_input("DCA 扣款日（每月）", min_value=1, max_value=28, value=1, step=1)
with col3:
    valuation_mode = st.radio(
        "估值日",
        options=["今天", "最後交易日 (max_date)"],
        index=0,
        help="決定比較表與曲線最後一天使用的價格與匯率。"
    )
valuation_to_today = (valuation_mode == "今天")

dca_amount_twd = st.number_input(
    "DCA 每月定額金額（台幣）", min_value=0, step=10000, value=70000,
    help="用於 DCA 比較分析的每月定額金額"
)

# ====== Run ======
if st.button("Run Comparison", type="primary"):
    with st.status("Running comparison...", expanded=False):
        try:
            result_core = run_full_comparison(
                df_input,
                dca_amount_twd=dca_amount_twd,
                mirror_list=compare_choices,
                dca_list=compare_choices,
                dca_day=dca_day,
                lumpsum_list=compare_choices,
                valuation_to_today=valuation_to_today
            )
            st.session_state["cmp_core"] = result_core
            st.session_state["compare_defaults"] = compare_choices
            st.success("Done!")
        except Exception as e:
            st.exception(e)
            st.stop()

core = st.session_state.get("cmp_core")
if core:
    meta = core["meta"]
    min_date, max_date, end_of_range = meta["min_date"], meta["max_date"], meta["end_of_range"]
    fx_data_dict = core["fx_data_dict"]
    stock_data_dict = core["stock_data_dict"]
    stock_close_daily = core["stock_close_daily"]
    fx_daily = core["fx_daily"]
    latest_prices = core["latest_prices"]
    base_index = core["base_index"]
    df_trades = core["df_trades"]

    # 先把 DCA 交易補上 comparison_trade_sets，再做估值
    comparison_trade_sets = list(core["comparison_trade_sets_init"])  # 鏡像+LumpSum
    for tgt in compare_choices:
        df_d = make_monthly_dca_trades(min_date, end_of_range, dca_amount_twd, tgt, fx_data_dict, stock_data_dict, dca_day=dca_day)
        if not df_d.empty:
            comparison_trade_sets.append((f"DCA-{tgt.replace('.TW','').replace('.T','')}", df_d))

    # 權益曲線 wide/long
    all_dates = pd.date_range(min_date, end_of_range, freq="D")
    eq_wide = base_index.rename(columns={"日期":"日期"}).copy()
    eq_long_parts = []

    # 你的投組（以 mirror 模擬器相同口徑跑一條 baseline：用原交易，但只估權益曲線）
    # 這裡使用 equity_curve_for_trades 以確保口徑與比較一致
    eq_base = equity_curve_for_trades(df_trades, all_dates, stock_close_daily, fx_daily, latest_prices, base_index).rename(columns={"權益(台幣)":"你的投組(平均成本法)"})
    eq_wide = eq_wide.merge(eq_base, on="日期", how="left")
    tmp_long = eq_base.rename(columns={"你的投組(平均成本法)":"權益(台幣)"})
    tmp_long["策略"] = "你的投組(平均成本法)"
    eq_long_parts.append(tmp_long[["日期","策略","權益(台幣)"]])

    # 其他策略曲線
    for label, df_like in comparison_trade_sets:
        curve = equity_curve_for_trades(df_like, all_dates, stock_close_daily, fx_daily, latest_prices, base_index)
        if not curve.empty:
            eq_wide = eq_wide.merge(curve.rename(columns={"權益(台幣)": label}), on="日期", how="left")
            c = curve.copy(); c["策略"] = label
            eq_long_parts.append(c[["日期","策略","權益(台幣)"]])

    eq_long = pd.concat(eq_long_parts, ignore_index=True) if eq_long_parts else pd.DataFrame(columns=["日期","策略","權益(台幣)"])

    # 產出概覽（以估值日終點快照）
    snapshots = []
    valuation_day = end_of_range
    final_row = eq_wide[eq_wide["日期"]==valuation_day]
    if not final_row.empty:
        for col in final_row.columns:
            if col == "日期": continue
            val = float(final_row.iloc[0][col]) if pd.notna(final_row.iloc[0][col]) else np.nan
            if pd.notna(val):
                snapshots.append({"策略": col, "權益(台幣)": round(val,0)})
    overview_df = pd.DataFrame(snapshots)

    # 明細報表彙整
    dataframes = {"comparison_equity_wide": eq_wide, "comparison_equity_long": eq_long, "comparison_overview": overview_df}
    for k, v in core["compare_sheets"].items():
        dataframes[k] = v

    st.subheader("多策略 vs 你的投組（概覽）")
    st.caption(f"估值日：{valuation_day.date()}")
    st.dataframe(dataframes["comparison_overview"], width="stretch")
    st.download_button(
        "comparison_overview.csv",
        dataframes["comparison_overview"].to_csv(index=False).encode("utf-8-sig"),
        "comparison_overview.csv",
        "text/csv"
    )

    st.markdown("---")
    st.subheader("📈 權益曲線比較（可多選）")
    eq_w = dataframes["comparison_equity_wide"]; eq_l = dataframes["comparison_equity_long"]
    if not eq_w.empty and not eq_l.empty:
        all_series = [c for c in eq_w.columns if c != "日期"]
        default_series = ["你的投組(平均成本法)"] + ([s for s in all_series if s != "你的投組(平均成本法)"][:1])
        picked = st.multiselect(
            "選擇要顯示的曲線",
            options=all_series,
            default=default_series,
            help="可多選。"
        )
        if picked:
            plot_df = eq_l[eq_l["策略"].isin(picked)].copy()
            try:
                import plotly.express as px
                fig_cmp = px.line(
                    plot_df,
                    x="日期", y="權益(台幣)", color="策略",
                    title="策略權益曲線比較（台幣）"
                )
                fig_cmp.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_cmp, use_container_width=True)
            except Exception:
                st.info("Plotly 無法載入，改以表格呈現。")
                st.dataframe(eq_w[["日期"] + picked], width="stretch")

            st.download_button(
                "comparison_equity_wide.csv",
                eq_w.to_csv(index=False).encode("utf-8-sig"),
                "comparison_equity_wide.csv",
                "text/csv"
            )
        else:
            st.warning("請至少勾選一條曲線顯示。")
    else:
        st.info("尚無可用的曲線資料。請先執行比較。")
