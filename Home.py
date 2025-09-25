import os
import streamlit as st
import requests
import pandas as pd


API_URL = "http://backend:8000"
# -------------------- Session --------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

# -------------------- Helper Functions --------------------
def api_post(endpoint, data=None, files=None):
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
    return requests.post(f"{API_URL}{endpoint}", data=data, files=files, headers=headers)

def api_get(endpoint):
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
    return requests.get(f"{API_URL}{endpoint}", headers=headers)

def api_delete(endpoint):
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
    return requests.delete(f"{API_URL}{endpoint}", headers=headers)

# -------------------- Sidebar for Login --------------------
st.sidebar.title("User Actions")

if st.sidebar.button("Logout"):
    st.session_state.token = None
    st.session_state.username = None

if not st.session_state.token:
    st.sidebar.subheader("Register / Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Register"):
        response = requests.post(f"{API_URL}/users", json={"username": username, "password": password})
        if response.status_code == 200:
            st.sidebar.success("User registered! You can login now.")
        else:
            st.sidebar.error(response.json().get("detail"))

    if st.sidebar.button("Login"):
        response = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.token = response.json()["access_token"]
            st.session_state.username = username
            st.sidebar.success(f"Logged in as {username}")
        else:
            st.sidebar.error("Login failed!")
else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")

# -------------------- Home Page Content --------------------
st.title("Portfolio Overview")

if st.session_state.token:
    st.subheader("Upload Portfolio CSV")
    portfolio_files = st.file_uploader("Select portfolio CSV files", type=["csv"], key="portfolio", accept_multiple_files=True)
    if portfolio_files:
        for file in portfolio_files:
            response = api_post("/users/me/trades/upload", files={"file": file})
            if response.status_code == 200:
                st.success(f"Uploaded {response.json()['count']} trades from {file.name}")
            else:
                st.error(f"Failed to upload {file.name}: {response.json().get('detail')}")

    # ----- Upload Portfolio JSON or XML -----
    st.subheader("Upload Portfolio JSON or XML")
    json_xml_files = st.file_uploader("Select JSON or XML files", type=["json", "xml"], key="json_xml", accept_multiple_files=True)
    if json_xml_files:
        for file in json_xml_files:
            response = api_post("/users/me/trades/upload_json_xml", files={"file": file})
            if response.status_code == 200:
                st.success(f"Uploaded from {file.name}: {response.json()}")
            else:
                st.error(f"Failed to upload {file.name}: {response.json().get('detail')}")

    # ----- Upload Historical CSV -----
    st.subheader("Upload Historical CSV")
    historical_files = st.file_uploader("Select historical CSV files", type=["csv"], key="historical", accept_multiple_files=True)
    if historical_files:
        for file in historical_files:
            response = api_post("/historical/upload", files={"file": file})
            if response.status_code == 200:
                st.success(f"Historical prices uploaded from {file.name}")
            else:
                st.error(f"Failed to upload {file.name}: {response.json().get('detail')}")

    # ----- Display Portfolio -----
    st.subheader("My Portfolio")
    response = api_get("/users/me/trades")
    df_trades = None
    if response.status_code == 200:
        trades = response.json()
        if trades:
            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades)
        else:
            st.info("No trades yet")
else:
    st.info("Please login to access the portfolio overview.")
