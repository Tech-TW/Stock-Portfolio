# ──────────────────────────────────────────────────────────
if uploaded_file.name.lower().endswith(".csv"):
df = _read_csv(uploaded_file)
else:
# Excel（若多工作表，可在此延伸為選擇 sheet）
maybe_df = _read_excel(uploaded_file)
if isinstance(maybe_df, dict):
first_key = list(maybe_df.keys())[0]
df = maybe_df[first_key]
else:
df = maybe_df
elif url_content is not None:
src_name = "remote_file"
# 嘗試自動判斷檔案格式（以 URL 副檔名或內容類型）
if url and (url.lower().endswith(".csv")):
df = _read_csv(url_content)
else:
try:
df = _read_excel(url_content)
except Exception:
# 回退試 CSV
df = _read_csv(url_content)


if df is not None:
st.success(f"已載入：{src_name}（{len(df)} 筆）")
st.dataframe(df.head(50), use_container_width=True)


# 存入 session_state，供其他頁使用
st.session_state["uploaded_df"] = df
st.info("✅ 檔案已載入；請切到左側（或上方）**🚀 執行分析** 頁面繼續。")
else:
st.warning("尚未載入檔案。請上傳或使用 URL 讀檔。")
