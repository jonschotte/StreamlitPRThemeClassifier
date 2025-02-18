import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# ==============================
# Model Selection Dropdown
# ==============================
model_name = st.selectbox(
    "Select Zero-Shot Classification Model:",
    ["facebook/bart-large-mnli", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"],
    index=0  # Default to BART
)

# Load Selected Model (with caching)
@st.cache_resource
def load_model(model_name):
    return pipeline("zero-shot-classification", model=model_name)

classifier = load_model(model_name)

# ==============================
# Function to Extract Article Text
# ==============================
def extract_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10, verify=False)  # Ignore SSL verification

        if response.status_code != 200:
            st.warning(f"Skipping {url}: HTTP {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract main text from <p> tags
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])

        return article_text.strip() if article_text else None

    except requests.exceptions.SSLError:
        st.error(f"SSL Error: Skipping {url}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
        return None

# ==============================
# Function to Classify Article Text
# ==============================
def classify_text(text, categories):
    if not text:  # Handle missing text
        return "Uncategorized"

    result = classifier(text[:1000], candidate_labels=categories)  # Truncate for performance
    return result["labels"][0]  # Return the category with highest confidence

# ==============================
# Streamlit App
# ==============================
st.title("🔍 URL Article Classifier")
st.write("Upload a CSV or Excel file with a `URL` column, and classify articles into custom categories.")
st.subheader(f"Using Model: `{model_name}`")

# ==============================
# Upload File (CSV or Excel)
# ==============================
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Detect file type and read accordingly
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        st.stop()

    # Standardize column names (strip spaces & make uppercase)
    df.columns = df.columns.str.strip().str.upper()

    # Ensure the file has a 'URL' column
    if "URL" not in df.columns:
        st.error("The uploaded file must contain a column named 'URL'.")
    else:
        st.success(f"File uploaded successfully! Detected {len(df)} URLs.")

        # ==============================
        # User Inputs for Categories
        # ==============================
        st.write("Enter categories separated by commas:")
        categories_input = st.text_area("Categories (comma-separated)",
                                        "Technology, Finance, Health, Sports, Entertainment")

        # Convert categories into a list
        categories = [c.strip() for c in categories_input.split(",") if c.strip()]

        # ==============================
        # Classify Articles
        # ==============================
        if st.button("Classify Articles"):
            with st.spinner("Processing URLs..."):
                df["Category"] = df["URL"].apply(
                    lambda url: classify_text(extract_text(url), categories) if pd.notna(url) else "Uncategorized")

            # ==============================
            # Show and Download Results
            # ==============================
            st.write("✅ Classification Completed!")
            st.dataframe(df[["URL", "Category"]])

            # Save results for download
            output_file = "classified_urls.xlsx"
            df.to_excel(output_file, index=False)

            # Download button for results
            with open(output_file, "rb") as f:
                st.download_button("Download Results", f, file_name="classified_urls.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.info("Switch between BART and DeBERTa models using the dropdown for comparison.")
