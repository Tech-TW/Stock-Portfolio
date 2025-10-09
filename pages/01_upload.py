import streamlit as st
import pandas as pd
from io import BytesIO

st.title("📥 Upload")

st.markdown("上傳 CSV/Excel，或以 URL（如 GitHub raw）讀檔。")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    key="file_uploader_main"
)

use_url = st.checkbox("Use URL instead (GitHub raw / cloud link)")
url_content = None

if use_url:
    url = st.text_input("Paste file URL (e.g., GitHub raw)")
    if st.button("Fetch file", use_container_width=True):
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

# 載入優先順序：本地上傳 > URL
df = None
src_name = None

if uploaded_file is not None:
    src_name = uploaded_file.name
    if uploaded_file.name.lower().endswith(".csv"):
        df = _read_csv(uploaded_file)
    else:
        maybe_df = _read_excel(uploaded_file)
        if isinstance(maybe_df, dict):
            first_key = list(maybe_df.keys())[0]
            df = maybe_df[first_key]
        else:
            df = maybe_df
elif url_content is not None:
    src_name = "remote_file"
    if "url" in locals() and url.lower().endswith(".csv"):
        df = _read_csv(url_content)
    else:
        try:
            df = _read_excel(url_content)
        except Exception:
            df = _read_csv(url_content)

if df is not None:
    st.success(f"Loaded: {src_name} ({len(df)} rows)")
    st.dataframe(df.head(50), use_container_width=True)
    st.session_state["uploaded_df"] = df
    st.info("✅ Data saved. Go to the **Analyze** page.")
else:
    st.warning("No file loaded yet. Upload or fetch via URL.")
