import pandas as pd
from langchain.llms import Ollama  # or your LLM wrapper

# Load file and extract columns
df = pd.read_excel(
    r"E:\Git-Repos\artificial-intelligence-latest\generative_ai\projects\DT_HDFC_Bank_Stmt_Aug_27_24_To_Aug_25_25.xls",
    sheet_name="Aug_27_24_To_Aug_25_25"
)
columns = df.columns.tolist()

# Format prompt
prompt = f"Here are the column names from a dataset: {columns}. Please explain what each column might represent."
numeric_cols = df.select_dtypes(include="number").columns.tolist()

sums = {col: df[col].sum() for col in numeric_cols}
prompt = f"The numeric columns and their sums are:\n"
for col, total in sums.items():
    prompt += f"- {col}: {total}\n"

print(prompt)

# Initialize model
llm = Ollama(model="gemma:2b")

# Get response
response = llm(prompt)
print(response)
