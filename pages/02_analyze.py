# ──────────────────────────────────────────────────────────
# 檔案：pages/02_🚀_執行分析.py
# 說明：執行分析、分頁顯示、下載報表（含：鏡像與 DCA 比較分析，可勾選標的）
# ──────────────────────────────────────────────────────────

# pages/02_analyze.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from collections import deque
from dateutil.relativedelta import relativedelta

import yfinance as yf

st.title("🚀 Analyze")

# 取得上傳資料
if "uploaded_df" not in st.session_state or st.session_state["uploaded_df"] is None:
    st.error("No data. Please upload at the **Upload** page first.")
    st.stop()

df_input: pd.DataFrame = st.session_state["uploaded_df"].copy()

# ========= 通用輔助 =========
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
    """下載指定幣別對台幣的歷史匯率 (Yahoo: XXXTWD=X)"""
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

# ========= 股票分割事件處理 =========
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
                        events["splits"].append({"date": r["Date"], "ratio": r["Split_Ratio"], "type": "split"})
        except Exception:
            pass
        self.events_data[ticker] = events
        return events

    def _calculate_shares_before_date(self, transaction_history: pd.DataFrame, ticker, target_date):
        t = transaction_history[
            (transaction_history["股票代號"]==ticker) &
            (transaction_history["日期"]<target_date)
        ]
        return max(0, float(t["購買股數"].sum()))

    def apply_stock_split_with_timing(self, position_data, ticker, split_date, split_ratio, transaction_history=None):
        if ticker not in position_data:
            return None
        pos = position_data[ticker]
        original_shares = pos["shares"]
        original_avg_cost = pos["avg_cost_foreign"]

        if transaction_history is not None:
            shares_before = self._calculate_shares_before_date(transaction_history, ticker, split_date)
            if shares_before <= 0:
                return None
            add_shares = shares_before * (split_ratio - 1.0)
            pos["shares"] = original_shares + add_shares
            total_cost_foreign = original_avg_cost * original_shares
            if pos["shares"]>0:
                pos["avg_cost_foreign"] = total_cost_foreign / pos["shares"]
        else:
            pos["shares"] *= split_ratio
            pos["avg_cost_foreign"] /= split_ratio

        if pos["shares"]>0 and pos["avg_cost_foreign"]>0:
            pos["total_cost_twd"] = pos["avg_cost_foreign"] * pos["shares"] * pos.get("avg_exchange_rate", pos.get("avg_fx", 1.0))
        return {
            "ticker": ticker, "date": split_date, "event":"stock_split", "ratio":split_ratio,
            "original_shares": original_shares, "new_shares": pos["shares"],
            "original_avg_cost": original_avg_cost, "new_avg_cost": pos["avg_cost_foreign"]
        }

    def process_all_splits_for_ticker(self, position_data, ticker, transaction_history=None):
        if ticker not in self.events_data: return []
        splits = self.events_data[ticker].get("splits", [])
        if not splits: return []
        splits_sorted = sorted(splits, key=lambda x: x["date"])
        out=[]
        for sp in splits_sorted:
            r = self.apply_stock_split_with_timing(position_data, ticker, sp["date"], sp["ratio"], transaction_history)
            if r: out.append(r)
        return out

# ========= FIFO（修正版，含交易成本） =========
def build_fifo_inventory_with_cost_fixed(df_trades, fx_data_dict, latest_prices=None):
    fifo_positions = {}
    fifo_realized_pnl_data = {}
    fifo_position_list = []

    for _, row in df_trades.sort_values("日期").iterrows():
        ticker = row["股票代號"]
        shares = float(row["購買股數"])
        price  = float(row["購買股價"])
        fx     = float(row.get("換匯匯率", 1.0))
        currency = row.get("幣別", determine_currency(ticker))
        fee    = float(row.get("交易成本", 0.0))

        fifo_positions.setdefault(ticker, deque())
        fifo_realized_pnl_data.setdefault(ticker, 0.0)

        if shares > 0:  # 買
            actual_cost_ps = (price * shares + fee) / shares
            fifo_positions[ticker].append({
                "shares": shares,
                "price": actual_cost_ps,      # 含交易成本
                "original_price": price,
                "fx": fx,
                "transaction_cost": fee,
                "currency": currency
            })
        else:          # 賣
            qty_to_sell = -shares
            realized = 0.0
            while qty_to_sell > 1e-12 and fifo_positions[ticker]:
                lot = fifo_positions[ticker][0]
                take = min(lot["shares"], qty_to_sell)
                per_share_fee = fee/qty_to_sell if qty_to_sell>0 else 0.0
                net_per_share = price - per_share_fee
                pnl_foreign = (net_per_share - lot["price"]) * take
                realized += pnl_foreign * lot["fx"]
                lot["shares"] -= take
                qty_to_sell   -= take
                if lot["shares"] <= 1e-12:
                    fifo_positions[ticker].popleft()
            fifo_realized_pnl_data[ticker] += realized

    for ticker, lots in fifo_positions.items():
        if not lots: continue
        currency = lots[0]["currency"]
        latest_fx = get_latest_fx_rate(currency, fx_data_dict)
        total_shares = sum(l["shares"] for l in lots)
        pure_cost_foreign_total = sum(l["original_price"]*l["shares"] for l in lots)
        total_cost_foreign      = sum(l["price"]*l["shares"] for l in lots)
        total_cost_twd          = sum(l["price"]*l["shares"]*l["fx"] for l in lots)
        avg_fx = (total_cost_twd/total_cost_foreign) if total_cost_foreign>0 else 1.0
        avg_cost = (total_cost_foreign/total_shares) if total_shares>0 else 0.0
        avg_cost_pure = (pure_cost_foreign_total/total_shares) if total_shares>0 else 0.0

        if latest_prices and ticker in latest_prices:
            last_px = float(latest_prices[ticker])
        else:
            try:
                data = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
                last_px = float(data["Close"].dropna().iloc[-1]) if not data.empty else np.nan
            except Exception:
                last_px = np.nan

        mv_foreign = 0.0 if np.isnan(last_px) else last_px*total_shares
        mv_twd     = mv_foreign * (latest_fx if not np.isnan(latest_fx) else 0.0)
        unreal_foreign = 0.0 if np.isnan(last_px) else (last_px-avg_cost)*total_shares
        unreal_twd     = unreal_foreign * (latest_fx if not np.isnan(latest_fx) else 0.0)
        total_unreal_twd = mv_twd - total_cost_twd
        fx_unreal_twd    = total_unreal_twd - unreal_twd

        fifo_position_list.append({
            "股票代號": ticker,
            "幣別": currency,
            "持有股數": total_shares,
            "平均成本(原幣)(未含交易成本)": avg_cost_pure,
            "平均成本(原幣)": avg_cost,
            "平均匯率成本": avg_fx,
            "總成本(原幣)": total_cost_foreign,
            "總成本(台幣)": total_cost_twd,
            "最新匯率": latest_fx,
            "現價(原幣)": last_px,
            "現價(台幣)": (last_px*latest_fx if not (np.isnan(last_px) or np.isnan(latest_fx)) else np.nan),
            "市值(原幣)": mv_foreign,
            "市值(台幣)": mv_twd,
            "未實現投資損益(原幣)": unreal_foreign,
            "未實現投資損益(台幣)": unreal_twd,
            "未實現總損益(台幣)": total_unreal_twd,
            "未實現投資匯率損益(台幣)": fx_unreal_twd
        })

    return pd.DataFrame(fifo_position_list), fifo_realized_pnl_data

# ===== 投資比較：輔助函式 =====
def get_price_on_or_before(date, ticker, stock_data_dict, min_date, max_date):
    """用既有 stock_data_dict 找 <= 指定日 最近一筆收盤價；若無則補抓一次。"""
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

def build_mirror_trade_row(trade_date, cash_twd, target_ticker, fx_data_dict, stock_data_dict, min_date, max_date):
    """回傳目標標的當日（或之前最近）價格與匯率（用於鏡像現金流換股數）。"""
    target_ccy = determine_currency(target_ticker)
    px = get_price_on_or_before(trade_date, target_ticker, stock_data_dict, min_date, max_date)
    fx = get_fx_rate(pd.to_datetime(trade_date), target_ccy, fx_data_dict)
    if np.isnan(px) or np.isnan(fx) or px <= 0 or fx <= 0:
        return None
    return {"_ok": True, "px": px, "fx": fx, "ccy": target_ccy}

def make_mirror_trades(df_trades, target_ticker, fx_data_dict, stock_data_dict, min_date, max_date):
    """用相同『台幣金額＋同日』把原現金流鏡像到指定目標標的。"""
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

        info = build_mirror_trade_row(d, cash_twd_abs * sign, target_ticker, fx_data_dict, stock_data_dict, min_date, max_date)
        if not info or not info.get("_ok", False):
            continue

        px_t, fx_t, ccy_t = info["px"], info["fx"], info["ccy"]
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
    """每月 dca_day 定期定額台幣 amount_twd ；遇假日取之前最近收盤價。"""
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

# ========= 主流程 =========
def run_full_analysis(trades_df: pd.DataFrame, dca_amount_twd: int = 70000,
                      mirror_list=None, dca_list=None, dca_day: int = 1) -> dict:
    # 1) 欄位準備
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

    # 2) 匯率＆股價歷史
    currencies = df_trades[df_trades["幣別"]!="TWD"]["幣別"].dropna().unique()
    fx_data_dict = {}
    for cur in currencies:
        f = download_fx_history(cur, min_date, max_date)
        if not f.empty:
            fx_data_dict[cur] = f
    tickers = df_trades["股票代號"].dropna().unique()
    stock_data_dict = {}
    for tkr in tickers:
        s = download_stock_history(tkr, min_date, max_date)
        if not s.empty:
            stock_data_dict[tkr] = s

    # 3) 更新「購買當時匯率」（歷史匯率）
    df_trades["購買當時匯率_歷史"] = df_trades.apply(
        lambda r: get_fx_rate(r["日期"], r["幣別"], fx_data_dict), axis=1
    )
    df_trades["購買當時匯率"] = df_trades["購買當時匯率_歷史"].fillna(df_trades["換匯匯率"])

    # 4) 平均成本法：持倉/已實現/交易成本
    position_data = {}
    realized_pnl_data = {}
    transaction_cost_data = {}

    for _, row in df_trades.sort_values("日期").iterrows():
        tkr = row["股票代號"]; sh = float(row["購買股數"])
        px  = float(row["購買股價"]); cur = row["幣別"]
        fx  = float(row["換匯匯率"]); fee = float(row["交易成本"])

        if tkr not in position_data:
            position_data[tkr] = {"shares":0.0,"avg_cost_foreign":0.0,"pure_cost_foreign_total":0.0,
                                  "avg_exchange_rate":0.0,"total_cost_twd":0.0,"currency":cur}
            realized_pnl_data[tkr] = {"total_realized_pnl":0.0,"currency":cur}
            transaction_cost_data[tkr] = {"total_cost":0.0,"currency":cur}

        pos = position_data[tkr]; realized = realized_pnl_data[tkr]; tx = transaction_cost_data[tkr]
        tx_twd = fee * fx
        tx["total_cost"] += tx_twd

        if sh > 0:
            actual_cost_ps = (px*sh + fee) / sh
            pos["pure_cost_foreign_total"] += px * sh
            new_shares = pos["shares"] + sh
            new_cost_foreign = pos["avg_cost_foreign"]*pos["shares"] + actual_cost_ps*sh
            new_cost_twd     = pos["total_cost_twd"] + actual_cost_ps*sh*fx
            if new_shares>0:
                pos["avg_cost_foreign"] = new_cost_foreign / new_shares
                pos["avg_exchange_rate"] = new_cost_twd / new_cost_foreign if new_cost_foreign>0 else 1.0
            pos["shares"] = new_shares
            pos["total_cost_twd"] = new_cost_twd
        else:
            sell = abs(sh)
            if position_data[tkr]["shares"] >= sell and position_data[tkr]["shares"]>0:
                gross = px*sell; net = gross - fee
                cost_basis_f = position_data[tkr]["avg_cost_foreign"]*sell
                real_f = (net - cost_basis_f)
                real_twd = real_f * position_data[tkr]["avg_exchange_rate"]
                realized["total_realized_pnl"] += real_twd
                cost_pure_foreign = (pos["pure_cost_foreign_total"]/pos["shares"]) * sell if pos["shares"]>0 else 0.0
                pos["pure_cost_foreign_total"] -= cost_pure_foreign
                pos["shares"] -= sell
                if pos["shares"]>0:
                    pos["total_cost_twd"] = pos["avg_cost_foreign"]*pos["shares"]*pos["avg_exchange_rate"]
                else:
                    pos["total_cost_twd"] = 0.0
                    pos["avg_cost_foreign"] = 0.0
                    pos["pure_cost_foreign_total"] = 0.0

    # 5) 股票分割（依時間）
    event_processor = StockEventProcessor()
    transaction_history = df_trades[["日期","股票代號","購買股數"]].copy()
    for tkr in position_data.keys():
        event_processor.fetch_stock_events(tkr, min_date, max_date)
        event_processor.process_all_splits_for_ticker(position_data, tkr, transaction_history)

    # 6) 生成庫存明細（平均成本口徑）
    position_list=[]
    latest_prices={}
    for tkr, pos in position_data.items():
        if pos["shares"]<=0: continue
        try:
            data = yf.download(tkr, period="5d", interval="1d", auto_adjust=True, progress=False)
            if not data.empty:
                latest_prices[tkr] = float(data["Close"].dropna().iloc[-1])
        except Exception:
            pass

    for tkr, pos in position_data.items():
        if pos["shares"]<=0: continue
        ccy = pos["currency"]
        last_px = latest_prices.get(tkr, np.nan)
        last_fx = get_latest_fx_rate(ccy, fx_data_dict)
        mv_foreign = 0.0 if np.isnan(last_px) else last_px*pos["shares"]
        mv_twd     = mv_foreign * (last_fx if not np.isnan(last_fx) else 0.0)
        unreal_invest_foreign = (0.0 if np.isnan(last_px) else (last_px - pos["avg_cost_foreign"])*pos["shares"])
        unreal_invest_twd     = unreal_invest_foreign * (last_fx if not np.isnan(last_fx) else 0.0)
        unreal_total_twd      = mv_twd - pos["total_cost_twd"]
        fx_unreal_twd         = unreal_total_twd - unreal_invest_twd

        position_list.append({
            "股票代號": tkr,
            "幣別": ccy,
            "持有股數": pos["shares"],
            "平均成本(原幣)(未含交易成本)": (pos["pure_cost_foreign_total"]/pos["shares"] if pos["shares"]>0 else 0.0),
            "平均成本(原幣)": pos["avg_cost_foreign"],
            "平均匯率成本": pos["avg_exchange_rate"],
            "總成本(原幣)": pos["avg_cost_foreign"]*pos["shares"],
            "總成本(台幣)": pos["total_cost_twd"],
            "最新匯率": last_fx,
            "現價(原幣)": last_px,
            "現價(台幣)": (last_px*last_fx if not (np.isnan(last_px) or np.isnan(last_fx)) else np.nan),
            "市值(原幣)": mv_foreign,
            "市值(台幣)": mv_twd,
            "未實現投資損益(原幣)": unreal_invest_foreign,
            "未實現投資損益(台幣)": unreal_invest_twd,
            "未實現總損益(台幣)": unreal_total_twd,
            "未實現投資匯率損益(台幣)": fx_unreal_twd
        })
    position_df = pd.DataFrame(position_list)

    # 7) 已實現損益、交易成本
    realized_df = pd.DataFrame([
        {"股票代號": t, "幣別": v["currency"], "已實現總損益(台幣)": v["total_realized_pnl"]}
        for t,v in realized_pnl_data.items() if abs(v["total_realized_pnl"])>1e-9
    ])
    cost_df = pd.DataFrame([
        {"股票代號": t, "幣別": v["currency"], "累計交易成本(台幣)": v["total_cost"]}
        for t,v in transaction_cost_data.items() if abs(v["total_cost"])>1e-9
    ])

    # 8) FIFO（修正版）
    fifo_position_df, fifo_realized_pnl_data = build_fifo_inventory_with_cost_fixed(
        df_trades, fx_data_dict, latest_prices
    )

    # 9) 明細表 display_df（含現價＆6欄損益）
    display_cols = ["日期","股票代號","幣別","購買股數","購買股價","換匯匯率","購買當時匯率","交易成本"]
    display_df = df_trades[[c for c in display_cols if c in df_trades.columns]].copy()
    for col in [
        "已實現總損益(台幣)", "已實現投資損益(台幣)", "已實現投資匯率損益(台幣)",
        "未實現總損益(台幣)", "未實現投資損益(台幣)", "未實現投資匯率損益(台幣)",
        "標的衡量日現價", "外匯現價"
    ]:
        display_df[col] = 0.0
    display_df["標的衡量日現價"] = display_df["股票代號"].map(latest_prices).astype(float)
    display_df["外匯現價"] = display_df["幣別"].apply(lambda c: float(get_latest_fx_rate(c, fx_data_dict)) if pd.notna(c) else np.nan)

    for ticker, g in df_trades.sort_values("日期").groupby("股票代號"):
        g = g.copy()
        currency   = g["幣別"].iloc[0]
        latest_px  = latest_prices.get(ticker, np.nan)
        latest_fx  = get_latest_fx_rate(currency, fx_data_dict)

        pool_shares = 0.0
        pool_avg_cost_foreign = 0.0
        pool_total_cost_foreign = 0.0
        pool_total_cost_twd = 0.0
        pool_avg_fx = 1.0
        buy_lots = []

        for idx, row in g.iterrows():
            shares = float(row["購買股數"])
            price  = float(row["購買股價"])
            fee    = float(row.get("交易成本", 0.0))
            fx     = float(row.get("換匯匯率", 1.0))

            if shares > 0:
                actual_cost_ps = (price*shares+fee)/shares
                new_total_shares       = pool_shares + shares
                new_total_cost_foreign = pool_total_cost_foreign + actual_cost_ps*shares
                new_total_cost_twd     = pool_total_cost_twd + actual_cost_ps*shares*fx
                pool_shares = new_total_shares
                pool_total_cost_foreign = new_total_cost_foreign
                pool_total_cost_twd     = new_total_cost_twd
                pool_avg_cost_foreign   = (pool_total_cost_foreign/pool_shares) if pool_shares>0 else 0.0
                pool_avg_fx             = (pool_total_cost_twd/pool_total_cost_foreign) if pool_total_cost_foreign>0 else 1.0
                buy_lots.append({"idx": idx, "remain": shares})
            else:
                sell_qty = -shares
                if sell_qty<=0 or pool_shares<=0:
                    continue
                per_share_fee = fee/sell_qty if sell_qty>0 else 0.0
                net_per_share_foreign = price - per_share_fee
                real_invest_foreign = (price - pool_avg_cost_foreign) * sell_qty
                real_total_foreign  = (net_per_share_foreign - pool_avg_cost_foreign) * sell_qty
                real_invest_twd = real_invest_foreign * pool_avg_fx
                real_total_twd  = real_total_foreign  * pool_avg_fx
                real_fx_twd     = real_total_twd - real_invest_twd

                display_df.loc[idx, "已實現投資損益(台幣)"] = real_invest_twd
                display_df.loc[idx, "已實現總損益(台幣)"]   = real_total_twd
                display_df.loc[idx, "已實現投資匯率損益(台幣)"] = real_fx_twd

                total_open = sum(l["remain"] for l in buy_lots)
                left = sell_qty
                for i, lot in enumerate(buy_lots):
                    if lot["remain"]<=0: continue
                    q = min(lot["remain"], sell_qty*(lot["remain"]/total_open)) if i<len(buy_lots)-1 else min(lot["remain"], left)
                    q = float(q)
                    lot["remain"] -= q
                    left -= q
                    if left<=1e-8: break

                pool_shares -= sell_qty
                if pool_shares <= 1e-8:
                    pool_shares = 0.0
                    pool_total_cost_foreign = 0.0
                    pool_total_cost_twd = 0.0
                    pool_avg_cost_foreign = 0.0
                    pool_avg_fx = 1.0
                else:
                    pool_total_cost_foreign = pool_avg_cost_foreign * pool_shares
                    pool_total_cost_twd     = pool_avg_cost_foreign * pool_shares * pool_avg_fx

        if (not np.isnan(latest_px)) and (not np.isnan(latest_fx)):
            for lot in buy_lots:
                remain = float(lot["remain"])
                if remain<=0: continue
                unreal_invest_twd = (latest_px - pool_avg_cost_foreign) * remain * latest_fx
                unreal_total_twd  = (latest_px * latest_fx * remain) - (pool_avg_cost_foreign * pool_avg_fx * remain)
                unreal_fx_twd     = unreal_total_twd - unreal_invest_twd
                bidx = lot["idx"]
                display_df.loc[bidx, "未實現投資損益(台幣)"]   += unreal_invest_twd
                display_df.loc[bidx, "未實現總損益(台幣)"]     += unreal_total_twd
                display_df.loc[bidx, "未實現投資匯率損益(台幣)"] += unreal_fx_twd

    # 10) 投組每日總彙整（到今天）
    today_tw = pd.Timestamp.today(tz="Asia/Taipei").normalize().tz_localize(None)
    all_dates = pd.date_range(min_date, today_tw, freq="D")
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

    if "預算餘額" in df.columns:
        cash_series = df[["日期","預算餘額"]].dropna(subset=["日期"]).copy()
        cash_series["日期"] = pd.to_datetime(cash_series["日期"]).dt.normalize()
        cash_series["預算餘額"] = pd.to_numeric(cash_series["預算餘額"], errors="coerce")
        cash_by_day = (
            cash_series.sort_values(["日期"])
            .drop_duplicates(subset=["日期"], keep="last")
            .set_index("日期")["預算餘額"]
            .reindex(all_dates).ffill().fillna(0.0)
        )
    else:
        cash_by_day = pd.Series(0.0, index=all_dates)

    positions = {}
    cum_realized_twd = 0.0
    trades_sorted = df_trades.sort_values("日期").copy()
    trades_sorted["日期"] = trades_sorted["日期"].dt.normalize()
    trades_by_day = {d:g for d,g in trades_sorted.groupby("日期")}
    last_day = all_dates[-1]

    def _latest_fx_safe(cur):
        v = get_latest_fx_rate(cur, fx_data_dict)
        if np.isnan(v):
            s = fx_daily.get(cur)
            return float(s.iloc[-1]) if s is not None and len(s) else (1.0 if cur=="TWD" else np.nan)
        return float(v)

    daily_rows=[]
    for day in all_dates:
        if day in trades_by_day:
            for _, r in trades_by_day[day].iterrows():
                tkr=r["股票代號"]; sh=float(r["購買股數"]); px=float(r["購買股價"])
                ccy=r["幣別"]; fx=float(r["換匯匯率"]); fee=float(r["交易成本"])
                if tkr not in positions:
                    positions[tkr] = {"shares":0.0,"avg_cost_foreign":0.0,"pure_cost_foreign_total":0.0,
                                      "avg_fx":0.0,"total_cost_twd":0.0,"currency":ccy}
                p=positions[tkr]
                if sh>0:
                    actual = (px*sh+fee)/sh
                    new_sh = p["shares"]+sh
                    new_cf = p["avg_cost_foreign"]*p["shares"] + actual*sh
                    new_ct = p["total_cost_twd"] + actual*sh*fx
                    if new_sh>0:
                        p["avg_cost_foreign"]= new_cf/new_sh
                        p["avg_fx"]= (new_ct/new_cf) if new_cf>0 else 1.0
                    p["shares"]=new_sh
                    p["total_cost_twd"]=new_ct
                    p["pure_cost_foreign_total"] += px*sh
                else:
                    sell = abs(sh)
                    if p["shares"]>=sell and p["shares"]>0:
                        gross = px*sell; net = gross - fee
                        real_f = (net - p["avg_cost_foreign"]*sell)
                        real_t = real_f * p["avg_fx"]
                        cum_realized_twd += real_t
                        cpp = (p["pure_cost_foreign_total"]/p["shares"])*sell if p["shares"]>0 else 0.0
                        p["pure_cost_foreign_total"] -= cpp
                        p["shares"] -= sell
                        if p["shares"]>0:
                            p["total_cost_twd"] = p["avg_cost_foreign"]*p["shares"]*p["avg_fx"]
                        else:
                            p["total_cost_twd"]=0.0; p["avg_cost_foreign"]=0.0; p["pure_cost_foreign_total"]=0.0

        total_mv_twd=0.0; total_cost_twd=0.0; unreal_invest_twd=0.0
        for tkr,p in positions.items():
            if p["shares"]<=0: continue
            ccy = p["currency"]
            if day==last_day:
                px_today = latest_prices.get(tkr, np.nan)
                if np.isnan(px_today):
                    px_today = float(stock_close_daily.get(tkr, pd.Series(index=all_dates)).get(day, np.nan))
                fx_today = _latest_fx_safe(ccy)
            else:
                px_today = float(stock_close_daily.get(tkr, pd.Series(index=all_dates)).get(day, np.nan))
                fx_today = float(fx_daily.get(ccy, pd.Series(index=all_dates)).get(day, np.nan))
            if np.isnan(px_today) or np.isnan(fx_today): continue
            mv_twd = px_today * p["shares"] * fx_today
            total_mv_twd += mv_twd
            total_cost_twd += p["total_cost_twd"]
            unreal_invest_twd += (px_today - p["avg_cost_foreign"]) * p["shares"] * fx_today

        unreal_total_twd = total_mv_twd - total_cost_twd
        unreal_fx_twd    = unreal_total_twd - unreal_invest_twd
        cash_twd = float(cash_by_day.get(day, 0.0))
        total_equity_twd = cum_realized_twd + total_mv_twd
        total_current_assets_twd = total_equity_twd + cash_twd

        daily_rows.append({
            "日期": day,
            "總流動資產(台幣)": round(total_current_assets_twd,0),
            "總權益(台幣)": round(total_equity_twd,0),
            "總市值(台幣)": round(total_mv_twd,0),
            "總成本(台幣)": round(total_cost_twd,0),
            "已實現損益(台幣)": round(cum_realized_twd,0),
            "未實現總損益(台幣)": round(unreal_total_twd,0),
            "未實現投資損益(台幣)": round(unreal_invest_twd,0),
            "未實現投資匯率損益(台幣)": round(unreal_fx_twd,0),
            "現金部位(台幣)": round(cash_twd,0)
        })
    daily_portfolio_df = pd.DataFrame(daily_rows).rename(columns={"總權益(台幣)":"投組總額_日報"})

    # 11) Summary
    total_twd_cost = float(position_df["總成本(台幣)"].sum()) if not position_df.empty else 0.0
    total_twd_value= float(position_df["市值(台幣)"].sum()) if not position_df.empty else 0.0
    total_unreal   = float(position_df["未實現總損益(台幣)"].sum()) if not position_df.empty else 0.0
    total_realized = float(realized_df["已實現總損益(台幣)"].sum()) if not realized_df.empty else 0.0
    summary = pd.DataFrame({
        "指標":["期間","總筆數","總投資成本(台幣)","市值(台幣)","未實現損益(台幣)","已實現損益(台幣)","總損益(台幣)","報酬率(%)"],
        "值":[f"{min_date.date()} ~ {max_date.date()}",
             len(df_trades), total_twd_cost, total_twd_value,
             total_unreal, total_realized, total_unreal+total_realized,
             round(((total_unreal+total_realized)/total_twd_cost*100.0), 2) if total_twd_cost>0 else np.nan]
    })

    # 12) 繪圖
    try:
        import plotly.express as px
        fig_equity = px.line(daily_portfolio_df, x="日期", y="投組總額_日報", title="投組總額-日報")
    except Exception:
        fig_equity = None

    dataframes = {
        "summary": summary,
        "trades": df_trades,
        "positions_avg": position_df,
        "positions_fifo": fifo_position_df,
        "realized": realized_df,
        "costs": cost_df,
        "daily_equity": daily_portfolio_df[["日期","投組總額_日報"]],
        "display_detail": display_df
    }

    # =====================================================
    # ＊比較分析（鏡像 + DCA）：動態標的（mirror_list / dca_list），可調 DCA 金額與扣款日
    # =====================================================
    comparison_results = []
    compare_sheets = {}

    # 預設標的（若使用者未選擇）
    default_targets = ["SPY", "0050.TW", "2330.TW"]
    mirror_targets = mirror_list if mirror_list else default_targets
    dca_targets    = dca_list if dca_list else default_targets

    def _evaluate_portfolio_fast(df_trades_like):
        pos = {}
        realized = {}
        for _, row in df_trades_like.sort_values("日期").iterrows():
            tkr = row["股票代號"]; sh = float(row["購買股數"]); px = float(row["購買股價"])
            fx  = float(row.get("換匯匯率", 1.0)); fee = float(row.get("交易成本", 0.0))
            ccy = row.get("幣別", determine_currency(tkr))
            if tkr not in pos:
                pos[tkr] = {"shares":0.0,"avg_cost_foreign":0.0,"avg_fx":0.0,"total_cost_twd":0.0,"currency":ccy}
                realized[tkr] = 0.0
            p = pos[tkr]
            if sh > 0:
                actual = (px*sh+fee)/sh
                new_sh = p["shares"] + sh
                new_cf = p["avg_cost_foreign"]*p["shares"] + actual*sh
                new_ct = p["total_cost_twd"] + actual*sh*fx
                p["shares"] = new_sh
                if new_sh>0:
                    p["avg_cost_foreign"] = new_cf/new_sh
                    p["avg_fx"]           = (new_ct/new_cf) if new_cf>0 else 1.0
                    p["total_cost_twd"]   = p["avg_cost_foreign"]*p["shares"]*p["avg_fx"]
                else:
                    p["avg_cost_foreign"]=0.0; p["avg_fx"]=1.0; p["total_cost_twd"]=0.0
            else:
                sell = abs(sh)
                if p["shares"] < sell or p["shares"]<=0: continue
                gross = px*sell; net = gross - fee
                total_foreign  = (net/sell - p["avg_cost_foreign"]) * sell
                total_twd      = total_foreign * p["avg_fx"]
                realized[tkr] += total_twd
                p["shares"] -= sell
                if p["shares"]>0:
                    p["total_cost_twd"] = p["avg_cost_foreign"]*p["shares"]*p["avg_fx"]
                else:
                    p["avg_cost_foreign"]=0.0; p["avg_fx"]=1.0; p["total_cost_twd"]=0.0

        rows=[]
        for tkr, p in pos.items():
            if p["shares"]<=0: continue
            latest_px = get_price_on_or_before(max_date, tkr, stock_data_dict, min_date, max_date)
            latest_fx = get_latest_fx_rate(p["currency"], fx_data_dict)
            mv_twd = latest_px * p["shares"] * latest_fx
            unreal_invest_twd = (latest_px - p["avg_cost_foreign"]) * p["shares"] * latest_fx
            unreal_total_twd  = mv_twd - p["total_cost_twd"]
            rows.append({
                "股票代號": tkr, "幣別": p["currency"], "持有股數": p["shares"],
                "平均成本(原幣)": p["avg_cost_foreign"], "平均匯率成本": p["avg_fx"],
                "總成本(台幣)": p["total_cost_twd"], "現價(原幣)": latest_px,
                "最新匯率": latest_fx, "市值(台幣)": mv_twd,
                "未實現投資損益(台幣)": unreal_invest_twd, "未實現總損益(台幣)": unreal_total_twd,
                "未實現投資匯率損益(台幣)": unreal_total_twd - unreal_invest_twd
            })
        position_df_alt = pd.DataFrame(rows).sort_values("股票代號")
        realized_total_twd = sum(realized.values())
        total_cost_twd = float(position_df_alt["總成本(台幣)"].sum()) if not position_df_alt.empty else 0.0
        total_mv_twd   = float(position_df_alt["市值(台幣)"].sum()) if not position_df_alt.empty else 0.0
        total_unreal_twd = float(position_df_alt["未實現總損益(台幣)"].sum()) if not position_df_alt.empty else 0.0
        total_pnl_twd  = realized_total_twd + total_unreal_twd
        total_return   = (total_pnl_twd / total_cost_twd) if total_cost_twd > 0 else np.nan
        summary_alt = {
            "總成本(台幣)": total_cost_twd, "市值(台幣)": total_mv_twd,
            "未實現損益(台幣)": total_unreal_twd, "已實現損益(台幣)": realized_total_twd,
            "總損益(台幣)": total_pnl_twd, "報酬率": total_return
        }
        return position_df_alt, realized, summary_alt

    # 鏡像（依勾選）
    for tgt in mirror_targets:
        df_m = make_mirror_trades(df_trades, tgt, fx_data_dict, stock_data_dict, min_date, max_date)
        if df_m.empty:
            continue
        pos_m, _, sum_m = _evaluate_portfolio_fast(df_m)
        disp_m = df_m.copy().sort_values("日期")
        disp_m["歷史匯率"] = disp_m["換匯匯率"]
        disp_m = disp_m[["日期","股票代號","幣別","購買股數","購買股價","換匯匯率","歷史匯率","交易成本"]]
        name = tgt.replace(".TW","").replace(".T","")
        compare_sheets[f"鏡像_{name}_買賣明細"] = disp_m
        compare_sheets[f"鏡像_{name}_庫存摘要"] = pos_m
        r = {"策略": f"鏡像-{name}"}; r.update(sum_m); comparison_results.append(r)

    # DCA（依勾選）
    for tgt in dca_targets:
        df_d = make_monthly_dca_trades(min_date, max_date, dca_amount_twd, tgt, fx_data_dict, stock_data_dict, dca_day=dca_day)
        if df_d.empty:
            continue
        pos_d, _, sum_d = _evaluate_portfolio_fast(df_d)
        disp_d = df_d.copy().sort_values("日期")
        disp_d["歷史匯率"] = disp_d["換匯匯率"]
        disp_d = disp_d[["日期","股票代號","幣別","購買股數","購買股價","換匯匯率","歷史匯率","交易成本"]]
        name = tgt.replace(".TW","").replace(".T","")
        compare_sheets[f"DCA_{name}_買賣明細"] = disp_d
        compare_sheets[f"DCA_{name}_庫存摘要"] = pos_d
        r = {"策略": f"DCA-{name}"}; r.update(sum_d); comparison_results.append(r)

    # 原投組 summary 放第一列
    base_summary = {
        "策略": "你的投組(平均成本法)",
        "總成本(台幣)": total_twd_cost,
        "市值(台幣)": total_twd_value,
        "未實現損益(台幣)": total_unreal,
        "已實現損益(台幣)": total_realized,
        "總損益(台幣)": total_unreal + total_realized,
        "報酬率": ( (total_unreal + total_realized) / total_twd_cost ) if total_twd_cost>0 else np.nan
    }
    if comparison_results:
        comparison_results.insert(0, base_summary)
        comparison_df = pd.DataFrame(comparison_results)[
            ["策略","總成本(台幣)","市值(台幣)","未實現損益(台幣)","已實現損益(台幣)","總損益(台幣)","報酬率"]
        ].copy()
        dataframes["comparison_overview"] = comparison_df
        for k, v in compare_sheets.items():
            dataframes[k] = v

    return {
        "meta":{"start":min_date,"end":max_date,"records":len(df_trades)},
        "dataframes": dataframes,
        "figures": {"equity_curve": fig_equity},
        "report_bytes": make_excel_report(dataframes)
    }

# ====== UI：比較標的與 DCA 參數 ======
st.divider()
col1, col2 = st.columns([2,1])
with col1:
    compare_choices = st.multiselect(
        "選擇比較標的（鏡像 + DCA）",
        options=["SPY", "0050.TW", "2330.TW"],
        default=["SPY", "0050.TW", "2330.TW"],
        help="鏡像與 DCA 都會使用這些標的。你可只選部分。"
    )
with col2:
    dca_day = st.number_input("DCA 扣款日（每月）", min_value=1, max_value=28, value=1, step=1)

dca_amount_twd = st.number_input(
    "DCA 每月定額金額（台幣）", min_value=0, step=10000, value=70000,
    help="用於 DCA 比較分析的每月定額金額"
)

# ====== Run ======
if st.button("Run Analysis", type="primary", use_container_width=True):
    with st.status("Running analysis...", expanded=False):
        try:
            result = run_full_analysis(
                df_input,
                dca_amount_twd=dca_amount_twd,
                mirror_list=compare_choices,
                dca_list=compare_choices,
                dca_day=dca_day
            )
            st.session_state["analysis_result"] = result
            st.success("Done!")
        except Exception as e:
            st.exception(e)
            st.stop()

result = st.session_state.get("analysis_result")
if result:
    dfs = result.get("dataframes", {})
    figs= result.get("figures", {})

    # 下載整包報表
    st.download_button(
        "Download Excel Report",
        data=result.get("report_bytes"),
        file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    tab_names = [
        "Summary", "Trades", "Positions (Avg)", "Positions (FIFO)",
        "Realized P/L", "Costs", "Daily Equity", "Detail (Buy/Sell)"
    ]
    if "comparison_overview" in dfs:
        tab_names.append("Comparisons")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader("Summary")
        st.dataframe(dfs["summary"], use_container_width=True)

    with tabs[1]:
        st.subheader("Trades")
        st.dataframe(dfs["trades"], use_container_width=True)
        st.download_button("trades.csv", dfs["trades"].to_csv(index=False).encode("utf-8-sig"), "trades.csv", "text/csv")

    with tabs[2]:
        st.subheader("Positions (Avg)")
        st.dataframe(dfs["positions_avg"], use_container_width=True)
        st.download_button("positions_avg.csv", dfs["positions_avg"].to_csv(index=False).encode("utf-8-sig"), "positions_avg.csv", "text/csv")

    with tabs[3]:
        st.subheader("Positions (FIFO)")
        st.dataframe(dfs["positions_fifo"], use_container_width=True)
        st.download_button("positions_fifo.csv", dfs["positions_fifo"].to_csv(index=False).encode("utf-8-sig"), "positions_fifo.csv", "text/csv")

    with tabs[4]:
        st.subheader("Realized P/L")
        st.dataframe(dfs["realized"], use_container_width=True)
        st.download_button("realized.csv", dfs["realized"].to_csv(index=False).encode("utf-8-sig"), "realized.csv", "text/csv")

    with tabs[5]:
        st.subheader("Costs")
        st.dataframe(dfs["costs"], use_container_width=True)
        st.download_button("costs.csv", dfs["costs"].to_csv(index=False).encode("utf-8-sig"), "costs.csv", "text/csv")

    with tabs[6]:
        st.subheader("Daily Equity / NAV Curve")
        st.dataframe(dfs["daily_equity"], use_container_width=True)
        if figs.get("equity_curve") is not None:
            st.plotly_chart(figs["equity_curve"], use_container_width=True)
        st.download_button("daily_equity.csv", dfs["daily_equity"].to_csv(index=False).encode("utf-8-sig"), "daily_equity.csv", "text/csv")

    with tabs[7]:
        st.subheader("Detail (Buy/Sell) with P/L Columns")
        st.dataframe(dfs["display_detail"], use_container_width=True)
        st.download_button("display_detail.csv", dfs["display_detail"].to_csv(index=False).encode("utf-8-sig"), "display_detail.csv", "text/csv")

    if "comparison_overview" in dfs:
        with tabs[-1]:
            st.subheader("六組策略 vs 你的投組（概覽）")
            st.dataframe(dfs["comparison_overview"], use_container_width=True)
            st.download_button(
                "comparison_overview.csv",
                dfs["comparison_overview"].to_csv(index=False).encode("utf-8-sig"),
                "comparison_overview.csv",
                "text/csv"
            )

