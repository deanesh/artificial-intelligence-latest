import pandas as pd
from langchain_community.llms import Ollama
import streamlit as st

# Step 1: Load and summarize Excel
def summarize_excel(file_path="data.xlsx"):
    file_path = r"E:\Git-Repos\artificial-intelligence-latest\generative_ai\projects\DT_HDFC_Bank_Stmt_Aug_27_24_To_Aug_25_25.xls"
    df = pd.read_excel(file_path)
    numeric_cols = df.select_dtypes(include='number').columns
    summary_text = "Here are the stats for each numeric column:\n"

    for col in numeric_cols:
        summary_text += (
            f"\n**{col}**:\n"
            f"- Sum: {df[col].sum():,.2f}\n"
            f"- Mean: {df[col].mean():,.2f}\n"
            f"- Min: {df[col].min():,.2f}\n"
            f"- Max: {df[col].max():,.2f}\n"
            f"- Std Dev: {df[col].std():,.2f}\n"
        )
    return summary_text

# Step 2: Pass to LLM
summary = summarize_excel("data.xlsx")

llm = Ollama(model="gemma:2b")

prompt = summary + "\n\nPlease analyze this data and suggest any patterns or anomalies you notice."

st.title("Excel Summary Assistant")

response = llm(prompt)
print("\n🤖 LLM Response:\n", response)

st.subheader("Summary:")
st.text(summary)

st.subheader("🤖 LLM Analysis:")
st.markdown(response)
