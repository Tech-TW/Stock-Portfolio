# ──────────────────────────────────────────────────────────
# 檔案：app.py（首頁）
# 說明：多頁 App 入口與全域設定（只在這裡 set_page_config）
# ──────────────────────────────────────────────────────────
# app.py
import streamlit as st

st.set_page_config(page_title="Portfolio Analyzer", page_icon="📊", layout="wide")

st.title("📊 Portfolio Analyzer (Multipage)")
st.markdown("""
歡迎！請依導覽執行：
1) 先到 **Upload** 頁上傳 CSV/Excel 或貼 GitHub raw 連結
2) 再到 **Analyze** 頁執行分析、瀏覽圖表與下載報表
""")

# 健康檢查 (方便你排錯)
with st.expander("Env / Debug info", expanded=False):
    try:
        import sys, platform, pathlib
        st.write("Python:", sys.version)
        st.write("Platform:", platform.platform())
        st.write("CWD:", pathlib.Path().resolve())
        import pandas, numpy, plotly, openpyxl, xlsxwriter, requests, chardet
        st.success("Imports OK")
    except Exception as e:
        st.error(f"Import error: {e}")

# 顯示是否已有上傳資料
has_df = "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None
st.info(f"Uploaded data in session: {'✅ Yes' if has_df else '❌ No'}")
if has_df:
    st.dataframe(st.session_state["uploaded_df"].head(20), use_container_width=True)
