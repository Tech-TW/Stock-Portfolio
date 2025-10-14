# ──────────────────────────────────────────────────────────
# 檔案：pages/02_🚀_執行分析.py
# 說明：執行分析、分頁顯示、下載報表（僅基礎投組分析；不含鏡像/DCA/Lump Sum）
# ──────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from collections import deque
from dateutil.relativedelta import relativedelta
import yfinance as yf

st.title("🚀 Analyze (基礎投組分析)")

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
                        events["splits"].append({"date": r["Date"], "ratio": r["Stock Splits"], "type": "split"})
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

# ========= 主流程（只做基礎分析）=========
def run_base_analysis(trades_df: pd.DataFrame, valuation_to_today: bool = True) -> dict:
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

    # 估值日設定
    today_tw = pd.Timestamp.today(tz="Asia/Taipei").normalize().tz_localize(None)
    end_of_range = today_tw if valuation_to_today else max_date
    valuation_day = end_of_range

    # 2) 匯率＆股價歷史（原投組）
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
        event_processor.fetch_stock_events(tkr, min_date, end_of_range)
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
        for t,v in (position_data and {k: realized_pnl_data[k] for k in realized_pnl_data} or {}).items()
        if abs(realized_pnl_data[k]["total_realized_pnl"])>1e-9
    ])
    # 如果上面的 list comprehension 因空 dict 造成 KeyError，保底：
    if realized_df.empty and any(realized_pnl_data.values()):
        realized_df = pd.DataFrame([
            {"股票代號": t, "幣別": v["currency"], "已實現總損益(台幣)": v["total_realized_pnl"]}
            for t,v in realized_pnl_data.items()
            if abs(v["total_realized_pnl"])>1e-9
        ])

    cost_df = pd.DataFrame([
        {"股票代號": t, "幣別": v["currency"], "累計交易成本(台幣)": v["total_cost"]}
        for t,v in transaction_cost_data.items() if abs(v["total_cost"])>1e-9
    ])

    # 8) FIFO（修正版）
    fifo_position_df, fifo_realized_pnl_data = build_fifo_inventory_with_cost_fixed(
        df_trades, fx_data_dict, latest_prices
    )

    # 9) 明細表 display_df（含現價＆6欄損益 placeholder）
    display_cols = ["日期","股票代號","幣別","購買股數","購買股價","換匯匯率","購買當時匯率","交易成本"]
    display_df = df_trades[[c for c in display_cols if c in df_trades.columns]].copy()
    for col in [
        "已實現總損益(台幣)", "已實現投資損益(台幣)", "已實現投資匯率損益(台幣)",
        "未實現總損益(台幣)", "未實現投資損益(台幣)", "未實現投資匯率損益(台幣)",
        "標的衡量日現價", "外匯現價"
    ]:
        display_df[col] = 0.0

    try:
        import plotly.express as px
        # 10) 投組每日總彙整（到估值日）＋曲線
        # 為簡化，僅用最新價估值的靜態 Summary 與明細，不額外跑日頻 NAV（需求在比較分頁）
        fig_equity = None
    except Exception:
        fig_equity = None

    # 11) Summary
    total_twd_cost  = float(position_df["總成本(台幣)"].sum()) if not position_df.empty else 0.0
    total_twd_value = float(position_df["市值(台幣)"].sum()) if not position_df.empty else 0.0
    total_unreal    = float(position_df["未實現總損益(台幣)"].sum()) if not position_df.empty else 0.0
    total_realized  = float(realized_df["已實現總損益(台幣)"].sum()) if not realized_df.empty else 0.0
    summary = pd.DataFrame({
        "指標":["期間","總筆數","總投資成本(台幣)","市值(台幣)","未實現損益(台幣)","已實現損益(台幣)","總損益(台幣)","報酬率(%)"],
        "值":[f"{min_date.date()} ~ {max_date.date()}",
             len(df_trades), total_twd_cost, total_twd_value,
             total_unreal, total_realized, total_unreal+total_realized,
             round(((total_unreal+total_realized)/total_twd_cost*100.0), 2) if total_twd_cost>0 else np.nan]
    })

    dataframes = {
        "summary": summary,
        "trades": df_trades,
        "positions_avg": position_df,
        "positions_fifo": fifo_position_df,
        "realized": realized_df,
        "costs": cost_df,
        "display_detail": display_df
    }

    return {
        "meta":{"start":min_date,"end":max_date,"records":len(df_trades),"valuation_day":valuation_day,"valuation_to_today":valuation_to_today},
        "dataframes": dataframes,
        "figures": {"equity_curve": fig_equity},
        "report_bytes": make_excel_report(dataframes)
    }

# ====== Run ======
valuation_mode = st.radio(
    "估值日", options=["今天", "最後交易日 (max_date)"], index=0,
    help="僅影響 summary/positions 等估值計算。"
)
valuation_to_today = (valuation_mode == "今天")

if st.button("Run Analysis", type="primary"):
    with st.status("Running analysis...", expanded=False):
        try:
            result = run_base_analysis(df_input, valuation_to_today=valuation_to_today)
            st.session_state["basic_analysis_result"] = result
            st.success("Done!")
        except Exception as e:
            st.exception(e)
            st.stop()

result = st.session_state.get("basic_analysis_result")
if result:
    dfs = result.get("dataframes", {})
    figs= result.get("figures", {})
    meta= result.get("meta", {})
    vday = meta.get("valuation_day")

    st.download_button(
        "Download Excel Report",
        data=result.get("report_bytes"),
        file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tab_names = ["Summary", "Trades", "Positions (Avg)", "Positions (FIFO)", "Realized P/L", "Costs", "Detail (Buy/Sell)"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader("Summary")
        st.dataframe(dfs["summary"], width="stretch")

    with tabs[1]:
        st.subheader("Trades")
        st.dataframe(dfs["trades"], width="stretch")
        st.download_button("trades.csv", dfs["trades"].to_csv(index=False).encode("utf-8-sig"), "trades.csv", "text/csv")

    with tabs[2]:
        st.subheader("Positions (Avg)")
        st.dataframe(dfs["positions_avg"], width="stretch")
        st.download_button("positions_avg.csv", dfs["positions_avg"].to_csv(index=False).encode("utf-8-sig"), "positions_avg.csv", "text/csv")

    with tabs[3]:
        st.subheader("Positions (FIFO)")
        st.dataframe(dfs["positions_fifo"], width="stretch")
        st.download_button("positions_fifo.csv", dfs["positions_fifo"].to_csv(index=False).encode("utf-8-sig"), "positions_fifo.csv", "text/csv")

    with tabs[4]:
        st.subheader("Realized P/L")
        st.dataframe(dfs["realized"], width="stretch")
        st.download_button("realized.csv", dfs["realized"].to_csv(index=False).encode("utf-8-sig"), "realized.csv", "text/csv")

    with tabs[5]:
        st.subheader("Costs")
        st.dataframe(dfs["costs"], width="stretch")
        st.download_button("costs.csv", dfs["costs"].to_csv(index=False).encode("utf-8-sig"), "costs.csv", "text/csv")

    with tabs[6]:
        st.subheader("Detail (Buy/Sell) with P/L Columns")
        st.dataframe(dfs["display_detail"], width="stretch")
        st.download_button("display_detail.csv", dfs["display_detail"].to_csv(index=False).encode("utf-8-sig"), "display_detail.csv", "text/csv")


