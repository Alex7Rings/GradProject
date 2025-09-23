import streamlit as st
import pandas as pd

from Home import api_get 

st.title("Grouped Portfolio")

if "token" in st.session_state and st.session_state.token:
    group_by = st.selectbox("Group by", ["instrument_type", "instrument_name"])
    response = api_get(f"/users/me/trades/grouped?group_by={group_by}")
    if response.status_code == 200:
        grouped_data = response.json()
        for group, trades in grouped_data.items():
            st.subheader(f"{group_by.capitalize().replace('_', ' ')}: {group}")
            df_group = pd.DataFrame(trades)
            st.dataframe(df_group)
    else:
        st.error("Failed to fetch grouped trades")
else:
    st.info("Please login to view grouped portfolio.")