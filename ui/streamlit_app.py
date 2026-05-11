import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(
    page_title="Application Review Analysis",
    page_icon="",
    layout="wide"
)


st.title("Application Review Analyzer")
st.markdown("")

#tabs section
tab1, tab2 = st.tabs(["Text Review", " Proceed with CSV"])

with tab1:

    st.subheader("Analyze Single Review")

    review_text = st.text_area("Enter your review:", height=150)

    if st.button("Analyze Review"):

        if not review_text.strip():
            st.warning("Empty review cannot be processed")
        else:
            with st.spinner("Cooking with attention"):

                response = requests.post(
                    API_URL,
                    json={"text": review_text}
                )

                if response.status_code == 200:
                    data = response.json()

                    result = data.get("data", data)
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Sentiment", result["sentiment"])
                    col2.metric("Urgency", result["urgency"])
                    col3.metric("Priority", result["priority"])

                    st.subheader("Detected Aspects")
                    st.write(", ".join(result["aspects"]))

                    st.subheader("Factors Influencing:")

                    words = result["explanation"]["top_words"]

                    for w in words:
                        st.write(f"**{w['word']}** (score: {w['score']:.4f})")

                else:
                    st.error("API is sleeping")


#csv

with tab2:

    st.subheader("CSV File Analysis")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df.head())

        if "reviews" not in df.columns:
            st.error("CSV must contain a column named 'reviews'")
        else:

            if st.button("Process CSV"):

                results = []
                progress_bar = st.progress(0)

                for i, review in enumerate(df["reviews"]):

                    response = requests.post(
                        API_URL,
                        json={"text": str(review)}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        result = data.get("data", data)

                        results.append({
                            "review": review,
                            "sentiment": result["sentiment"],
                            "urgency": result["urgency"],
                            "priority": result["priority"],
                            "aspects": ", ".join(result["aspects"]),
                            "top_words": ", ".join(
                                [w["word"] for w in result["explanation"]["top_words"]]
                            )
                        })

                    progress_bar.progress((i + 1) / len(df))

                result_df = pd.DataFrame(results)

                st.success("CSV processed Sucessfully")
                st.dataframe(result_df.head())
                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Processed CSV",
                    data=csv,
                    file_name="processed_reviews.csv",
                    mime="text/csv"
                )
