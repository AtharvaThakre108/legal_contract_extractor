import streamlit as st
import requests
import json

st.set_page_config(page_title="Legal Contract Analyzer", layout="wide")
st.title("Legal Contract Clause Extractor")
st.write("Upload a contract PDF to extract and summarize key legal clauses.")

uploaded_file = st.file_uploader("Upload Contract PDF", type=['pdf'])

RISK_COLORS = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}

if uploaded_file:
    with st.spinner("Analyzing contract..."):
        response = requests.post(
            "http://127.0.0.1:8000/extract",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )

    if response.status_code == 200:
        data = response.json()
        st.success(f"Found {data['total_clauses_found']} clauses in {data['filename']}")

        for clause in data['clauses']:
            risk = clause['risk_level']
            with st.expander(f"{RISK_COLORS[risk]} {clause['clause_type']} — {risk} RISK"):
                st.markdown("**Plain English Summary:**")
                st.write(clause['summary'])
                st.markdown("**Original Clause Text:**")
                st.text_area("", clause['original_text'], height=150,
                           key=clause['clause_type'])
    else:
        st.error(f"Error: {response.text}")