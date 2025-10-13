# pages/03_📦_持倉上傳分析.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import timedelta

import yfinance as yf

st.title("📦 持倉上傳分析（無交易明細）")

st.markdown(
    """
上傳目前投資組合持倉，欄位需包含：

- **股票代號**、**幣別**（可留空，會自動推斷）、**持有股數**、  
- **平均成本(原幣)(未含交易成本)**、**平均成本(原幣)**、**平均匯率成本**

> 本頁不需要交易明細；會直接以持倉平均成本做**平均成本法**與（等價呈現的）**FIFO**庫存計算，並抓最新價格/匯率產生市值與損益。
"""
)

# ========== 與你主程式一致的輔助 ==========
def determine_currency(ticker: str) -> str:
    t = str(ticker).upper()
    if t.endswith(".TW") or t.endswith(".TWO"):
        return "TWD"
    if t.isdigit() and len(t) == 4:
        return "TWD"
    if t.endswith(".HK"):
        return "HKD"
    if t.endswith(".T"):
        return "JPY"
    if t.endswith(".L"):
        return "GBP"
    if t.endswith(".DE") or t.endswith(".F"):
        return "EUR"
    if t.endswith(".TO"):
        return "CAD"
    if t.endswith(".AX"):
        return "AUD"
    return "USD"

def download_fx_history(currency: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if currency == "TWD":
        return pd.DataFrame({"日期": [start_date], "匯率": [1.0], "幣別": ["TWD"]})
    try:
        fx_symbol = f"{currency}TWD=X"
        start_ex = start_date - timedelta(days=10)
        end_ex = end_date + timedelta(days=10)
        fx = yf.download(fx_symbol, start=start_ex, end=end_ex, auto_adjust=True, progress=False)
        if fx.empty:
            return pd.DataFrame()
        fx_df = fx["Close"].reset_index()
        fx_df.columns = ["日期", "匯率"]
        fx_df["日期"] = pd.to_datetime(fx_df["日期"]).dt.normalize()
        fx_df["幣別"] = currency
        return fx_df
    except Exception:
        return pd.DataFrame()

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
    if f2.empty:
        return np.nan
    return float(f2.iloc[0]["匯率"])

def download_stock_history(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        start_ex = start_date - timedelta(days=10)
        end_ex = end_date + timedelta(days=10)
        s = yf.download(ticker, start=start_ex, end=end_ex, auto_adjust=True, progress=False)
        if s.empty:
            return pd.DataFrame()
        sdf = s["Close"].reset_index()
        sdf.columns = ["日期", "收盤價"]
        sdf["日期"] = pd.to_datetime(sdf["日期"]).dt.normalize()
        sdf["股票代號"] = ticker
        return sdf
    except Exception:
        return pd.DataFrame()

# ========== 上傳區（與 Upload 頁相同模式） ==========
uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    key="holdings_file_uploader",
)

use_url = st.checkbox("Use URL instead (GitHub raw / cloud link)", key="holdings_use_url")
url_content = None
if use_url:
    url = st.text_input("Paste file URL (e.g., GitHub raw)", key="holdings_url")
    if st.button("Fetch file", use_container_width=True, key="holdings_fetch"):
        try:
            import requests
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            url_content = r.content
            st.success("URL fetched successfully.")
        except Exception as e:
            st.error(f"URL fetch failed: {e}")
            url_content = None

@st.cache_data(show_spinner=False)
def _read_csv(file_or_bytes) -> pd.DataFrame:
    try:
        import chardet
        if isinstance(file_or_bytes, (bytes, bytearray)):
            enc = chardet.detect(file_or_bytes).get("encoding") or "utf-8"
            return pd.read_csv(BytesIO(file_or_bytes), encoding=enc)
        return pd.read_csv(file_or_bytes)
    except Exception:
        if isinstance(file_or_bytes, (bytes, bytearray)):
            return pd.read_csv(BytesIO(file_or_bytes), encoding="utf-8")
        return pd.read_csv(file_or_bytes, encoding="utf-8")

@st.cache_data(show_spinner=False)
def _read_excel(file_or_bytes) -> pd.DataFrame:
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return pd.read_excel(BytesIO(file_or_bytes))
    return pd.read_excel(file_or_bytes)

def _load_df(uploaded_file, url_content):
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            return _read_csv(uploaded_file), uploaded_file.name
        else:
            maybe_df = _read_excel(uploaded_file)
            if isinstance(maybe_df, dict):
                first_key = list(maybe_df.keys())[0]
                return maybe_df[first_key], uploaded_file.name
            return maybe_df, uploaded_file.name
    elif url_content is not None:
        try:
            if "url" in locals() and str(url).lower().endswith(".csv"):
                return _read_csv(url_content), "remote_file.csv"
            try:
                return _read_excel(url_content), "remote_file.xlsx"
            except Exception:
                return _read_csv(url_content), "remote_file.csv"
        except Exception as e:
            st.error(f"Read failed: {e}")
            return None, None
    return None, None

df_holdings, src_name = _load_df(uploaded_file, url_content)

if df_holdings is None:
    st.info("請上傳目前持倉表。")
    st.stop()

st.success(f"Loaded: {src_name} ({len(df_holdings)} rows)")
st.dataframe(df_holdings.head(50), use_container_width=True)

# ========== 欄位檢查與清理 ==========
required_cols = [
    "股票代號",
    "幣別",
    "持有股數",
    "平均成本(原幣)(未含交易成本)",
    "平均成本(原幣)",
    "平均匯率成本",
]

missing = [c for c in required_cols if c not in df_holdings.columns]
if missing:
    st.error(f"缺少必要欄位：{missing}")
    st.stop()

df = df_holdings.copy()

# 幣別自動補推斷
df["幣別"] = df["幣別"].fillna(df["股票代號"].apply(determine_currency))

# 型別清理
for numcol in ["持有股數", "平均成本(原幣)(未含交易成本)", "平均成本(原幣)", "平均匯率成本"]:
    df[numcol] = pd.to_numeric(df[numcol], errors="coerce").fillna(0.0)

# 允許選擇用含或不含交易成本的平均成本
cost_choice = st.radio(
    "平均成本使用：",
    ["平均成本(原幣)", "平均成本(原幣)(未含交易成本)"],
    index=0,
    horizontal=True,
)
cost_col = cost_choice

# ========== 抓價抓匯 ==========
# 找一個適用的日期範圍（這一頁只用「最新」即可，但 fx 抓資料需起訖）
today = pd.Timestamp.today(tz="Asia/Taipei").normalize().tz_localize(None)
start_date = today - pd.Timedelta(days=10)
end_date = today

fx_data_dict = {}
currencies = df["幣別"].dropna().unique()
for cur in currencies:
    fx_df = download_fx_history(cur, start_date, end_date)
    if not fx_df.empty:
        fx_data_dict[cur] = fx_df

latest_fx_map = {cur: get_latest_fx_rate(cur, fx_data_dict) for cur in currencies}

latest_px_map = {}
for tkr in df["股票代號"].dropna().unique():
    try:
        data = yf.download(tkr, period="5d", interval="1d", auto_adjust=True, progress=False)
        latest_px_map[tkr] = float(data["Close"].dropna().iloc[-1]) if not data.empty else np.nan
    except Exception:
        latest_px_map[tkr] = np.nan

# ========== 平均成本法庫存表 ==========
avg_rows = []
for _, r in df.iterrows():
    tkr = str(r["股票代號"])
    ccy = str(r["幣別"])
    shares = float(r["持有股數"])
    avg_cost_foreign = float(r[cost_col])            # 依選擇使用含/不含交易成本的平均成本
    avg_fx = float(r["平均匯率成本"]) if r["平均匯率成本"] else latest_fx_map.get(ccy, np.nan)

    last_px = latest_px_map.get(tkr, np.nan)
    last_fx = latest_fx_map.get(ccy, np.nan)

    total_cost_foreign = avg_cost_foreign * shares
    total_cost_twd = total_cost_foreign * (avg_fx if not np.isnan(avg_fx) else 0.0)

    mv_foreign = 0.0 if np.isnan(last_px) else last_px * shares
    mv_twd = mv_foreign * (last_fx if not np.isnan(last_fx) else 0.0)

    unreal_invest_foreign = 0.0 if np.isnan(last_px) else (last_px - avg_cost_foreign) * shares
    unreal_invest_twd = unreal_invest_foreign * (last_fx if not np.isnan(last_fx) else 0.0)
    unreal_total_twd = mv_twd - total_cost_twd
    fx_unreal_twd = unreal_total_twd - unreal_invest_twd

    # 估回未含交易成本平均成本（若選的是含交易成本，則補一欄）
    avg_cost_pure = float(r["平均成本(原幣)(未含交易成本)"])

    avg_rows.append({
        "股票代號": tkr,
        "幣別": ccy,
        "持有股數": shares,
        "平均成本(原幣)(未含交易成本)": avg_cost_pure,
        "平均成本(原幣)": avg_cost_foreign,
        "平均匯率成本": avg_fx,
        "總成本(原幣)": total_cost_foreign,
        "總成本(台幣)": total_cost_twd,
        "最新匯率": last_fx,
        "現價(原幣)": last_px,
        "現價(台幣)": (last_px * last_fx if not (np.isnan(last_px) or np.isnan(last_fx)) else np.nan),
        "市值(原幣)": mv_foreign,
        "市值(台幣)": mv_twd,
        "未實現投資損益(原幣)": unreal_invest_foreign,
        "未實現投資損益(台幣)": unreal_invest_twd,
        "未實現總損益(台幣)": unreal_total_twd,
        "未實現投資匯率損益(台幣)": fx_unreal_twd,
    })

positions_avg = pd.DataFrame(avg_rows)

st.subheader("📊 平均成本法庫存明細")
st.dataframe(positions_avg, use_container_width=True)

# ========== FIFO 庫存表（無賣出 → 與平均成本結果一致；但保留同欄位結構） ==========
positions_fifo = positions_avg.copy()
positions_fifo = positions_fifo[
    [
        "股票代號", "幣別", "持有股數",
        "平均成本(原幣)(未含交易成本)", "平均成本(原幣)", "平均匯率成本",
        "總成本(原幣)", "總成本(台幣)",
        "最新匯率", "現價(原幣)", "現價(台幣)",
        "市值(原幣)", "市值(台幣)",
        "未實現投資損益(原幣)", "未實現投資損益(台幣)",
        "未實現總損益(台幣)", "未實現投資匯率損益(台幣)",
    ]
].copy()

st.subheader("📚 FIFO 庫存明細（僅持有、無賣出 → 結果與平均成本一致）")
st.dataframe(positions_fifo, use_container_width=True)

# ========== Summary 與每日總額（以今日為基準） ==========
total_twd_cost = float(positions_avg["總成本(台幣)"].sum()) if not positions_avg.empty else 0.0
total_twd_value = float(positions_avg["市值(台幣)"].sum()) if not positions_avg.empty else 0.0
total_unreal = float(positions_avg["未實現總損益(台幣)"].sum()) if not positions_avg.empty else 0.0

summary = pd.DataFrame(
    {
        "指標": [
            "日期",
            "總投資成本(台幣)",
            "市值(台幣)",
            "未實現損益(台幣)",
            "報酬率(%)",
        ],
        "值": [
            f"{today.date()}",
            round(total_twd_cost, 0),
            round(total_twd_value, 0),
            round(total_unreal, 0),
            round((total_unreal / total_twd_cost * 100.0), 2) if total_twd_cost > 0 else np.nan,
        ],
    }
)

st.subheader("🧾 Summary")
st.dataframe(summary, use_container_width=True)

daily_equity = pd.DataFrame(
    {"日期": [today], "投組總額_日報": [round(total_twd_value, 0)]}
)

# ========== 可視化（簡版） ==========
try:
    import plotly.express as px

    st.subheader("📈 投組總額（今日）")
    fig = px.line(daily_equity, x="日期", y="投組總額_日報", title="投組總額-日報（今日）")
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    pass

# ========== 匯出 Excel ==========
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

export_pack = {
    "summary": summary,
    "positions_avg": positions_avg,
    "positions_fifo": positions_fifo,
    "daily_equity": daily_equity,
    "input_holdings": df_holdings,
}

bin_xlsx = make_excel_report(export_pack)
st.download_button(
    "⬇️ 下載報表（Excel）",
    data=bin_xlsx,
    file_name="holdings_quick_analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
