import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load Hugging Face text classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# Function to extract article text using BeautifulSoup
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


# Function to classify text using Hugging Face model
def classify_text(text, categories):
    if not text:  # Handle missing text
        return "Uncategorized"

    result = classifier(text[:1000], candidate_labels=categories)  # Truncate for performance
    return result["labels"][0]  # Return the category with highest confidence


# Streamlit App
st.title("🔍 URL Article Classifier")
st.write("Upload a CSV or Excel file with a `URL` column, and classify articles into custom categories.")

# Upload File (CSV or Excel)
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

    # Ensure the file has a 'URL' column
    if "URL" not in df.columns:
        st.error("The uploaded file must contain a column named 'URL'.")
    else:
        st.success(f"File uploaded successfully! Detected {len(df)} URLs.")

        # Allow user to enter categories
        st.write("Enter categories separated by commas:")
        categories_input = st.text_area("Categories (comma-separated)",
                                        "Technology, Finance, Health, Sports, Entertainment")

        # Convert categories into a list
        categories = [c.strip() for c in categories_input.split(",") if c.strip()]

        if st.button("Classify Articles"):
            with st.spinner("Processing URLs..."):
                df["Category"] = df["URL"].apply(
                    lambda url: classify_text(extract_text(url), categories) if pd.notna(url) else "Uncategorized")

            # Show results
            st.write("✅ Classification Completed!")
            st.dataframe(df[["URL", "Category"]])

            # Save results for download
            output_file = "classified_urls.xlsx"
            df.to_excel(output_file, index=False)

            # Download button for results
            with open(output_file, "rb") as f:
                st.download_button("Download Results", f, file_name="classified_urls.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
