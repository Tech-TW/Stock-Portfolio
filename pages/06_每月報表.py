# ──────────────────────────────────────────────────────────
# 檔案：pages/06_每月報表.py
# 說明：專門顯示／匯出「每月月底」的股票總成本＆總市值報表
# 依賴：請先在「02_🚀_執行分析 / Analyze」頁面執行分析，該頁會把 monthly_eom 放進 session_state
# ──────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

st.title("📅 每月報表（月底口徑）")

# 1) 檢查是否已有分析結果（由 02_🚀_執行分析.py 寫入）
res = st.session_state.get("analysis_result")
if not res or "dataframes" not in res:
    st.error("尚未找到分析結果。請先到 **🚀 Analyze** 頁面執行分析。")
    st.stop()

dfs = res["dataframes"]
if "monthly_eom" not in dfs or dfs["monthly_eom"] is None or dfs["monthly_eom"].empty:
    st.error("沒有找到『Monthly EoM』資料。請先到 **🚀 Analyze** 頁面按下 Run Analysis。")
    st.stop()

monthly_eom: pd.DataFrame = dfs["monthly_eom"].copy()
monthly_eom = monthly_eom.sort_values("月底").reset_index(drop=True)

# 2) 篩選器（可選）— 依日期區間過濾
st.subheader("篩選條件（可選）")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("起始月份（含）", value=monthly_eom["月底"].min().date() if not monthly_eom.empty else None)
with col2:
    end_date = st.date_input("結束月份（含）", value=monthly_eom["月底"].max().date() if not monthly_eom.empty else None)

if start_date and end_date:
    _mask = (monthly_eom["月底"].dt.date >= start_date) & (monthly_eom["月底"].dt.date <= end_date)
    monthly_view = monthly_eom.loc[_mask].copy()
else:
    monthly_view = monthly_eom.copy()

st.divider()

# 3) 顯示資料表
st.subheader("每月月底報表（股票總成本 / 總市值）")
st.caption("口徑與 **🚀 Analyze** 產出的每日投組口徑一致（估值日邏輯沿用該頁）。")
st.dataframe(monthly_view, use_container_width=True)

# 4) 匯出：CSV
st.download_button(
    "下載 monthly_eom.csv",
    monthly_view.to_csv(index=False).encode("utf-8-sig"),
    file_name="monthly_eom.csv",
    mime="text/csv",
    use_container_width=True
)

# 5) 匯出：只含月報的 Excel
def _to_excel_only_monthly(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        safe = "monthly_eom"[:31]
        df.to_excel(writer, sheet_name=safe, index=False)
    buf.seek(0)
    return buf.read()

st.download_button(
    "下載 Excel（僅每月報表）",
    data=_to_excel_only_monthly(monthly_view),
    file_name=f"monthly_eom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

# 6) 視覺化（可選）
st.subheader("趨勢圖（可選）")
try:
    import plotly.express as px
    # 長表 for 折線（兩條：成本 vs 市值）
    mlong = monthly_view.melt(
        id_vars=["月底"],
        value_vars=["股票總成本(台幣)", "股票總市值(台幣)"],
        var_name="指標", value_name="金額(台幣)"
    )
    if not mlong.empty:
        fig = px.line(mlong, x="月底", y="金額(台幣)", color="指標", markers=True,
                      title="每月月底：總成本 vs 總市值（台幣）")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("目前篩選範圍內沒有資料。")
except Exception:
    st.info("Plotly 無法載入，僅顯示表格與下載。")

st.success("完成！你也可以回到 **🚀 Analyze** 下載整包 Excel（含 monthly_eom 工作表）。")
