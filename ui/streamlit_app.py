import streamlit as st
import requests
import pandas as pd


# API URL

API_URL = "https://app-review-sentiment-and-urgency-analysis.onrender.com/analyze"


# Page Config
st.set_page_config(
    page_title="Application Review Analysis",
    page_icon="📱",
    layout="wide"
)


# Title
st.title("📱 Application Review Analyzer")

st.markdown(
    "Analyze app reviews for sentiment, urgency, "
    "priority and key influencing words."
)


# Tabs
tab1, tab2 = st.tabs([
    "Text Review",
    "Proceed with CSV"
])


with tab1:

    st.subheader("Analyze Single Review")

    review_text = st.text_area(
        "Enter your review:",
        height=150
    )

    if st.button("Analyze Review"):

        if not review_text.strip():

            st.warning(
                "Empty review cannot be processed."
            )

        else:

            with st.spinner(
                "Analyzing review..."
            ):

                try:

                    response = requests.post(
                        API_URL,
                        json={
                            "text": review_text
                        },
                        timeout=60
                    )

                    # API Success
                    if response.status_code == 200:

                        response_json = response.json()

                        if not response_json.get(
                            "success",
                            False
                        ):

                            st.error(
                                "Prediction failed."
                            )

                        else:

                            result = response_json["data"]

                            # Metrics
                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Sentiment",
                                result["sentiment"]
                            )

                            col2.metric(
                                "Urgency",
                                result["urgency"]
                            )

                            col3.metric(
                                "Priority",
                                result["priority"]
                            )

                            # Aspects
                            st.subheader(
                                "Detected Aspects"
                            )

                            st.write(
                                ", ".join(
                                    result["aspects"]
                                )
                            )

                            # Explanation
                            st.subheader(
                                "Important Influencing Words"
                            )

                            words = result[
                                "explanation"
                            ]["top_words"]

                            for w in words:

                                st.write(
                                    f"**{w['word']}** "
                                    f"(score: "
                                    f"{w['score']:.4f})"
                                )

                    # API Failure
                    else:

                        st.error(
                            f"API Error: "
                            f"{response.status_code}"
                        )

                except requests.exceptions.Timeout:

                    st.error(
                        "Request timed out. "
                        "Backend may be waking up."
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to backend API."
                    )

                except Exception as e:

                    st.error(
                        f"Unexpected Error: {e}"
                    )


with tab2:

    st.subheader("CSV File Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.write("Preview:")

            st.dataframe(df.head())

            # Validate
            if "reviews" not in df.columns:

                st.error(
                    "CSV must contain a "
                    "'reviews' column."
                )

            else:

                if st.button("Process CSV"):

                    results = []

                    progress_bar = st.progress(0)

                    total_reviews = len(df)

                    for i, review in enumerate(df["reviews"]):

                        try:

                            response = requests.post(
                                API_URL,
                                json={
                                    "text": str(review)
                                },
                                timeout=60
                            )

                            if response.status_code == 200:

                                response_json = response.json()

                                if response_json.get(
                                    "success",
                                    False
                                ):

                                    result = response_json[
                                        "data"
                                    ]

                                    results.append({

                                        "review": review,

                                        "sentiment":
                                        result["sentiment"],

                                        "urgency":
                                        result["urgency"],

                                        "priority":
                                        result["priority"],

                                        "aspects":
                                        ", ".join(
                                            result["aspects"]
                                        ),

                                        "top_words":
                                        ", ".join([
                                            w["word"]
                                            for w in result[
                                                "explanation"
                                            ]["top_words"]
                                        ])
                                    })

                        except Exception:

                            results.append({

                                "review": review,

                                "sentiment": "Error",

                                "urgency": "Error",

                                "priority": "Error",

                                "aspects": "Error",

                                "top_words": "Error"
                            })

                        progress_bar.progress(
                            (i + 1) / total_reviews
                        )

                    # Results DataFrame
                    result_df = pd.DataFrame(results)

                    st.success(
                        "CSV processed successfully!"
                    )

                    st.dataframe(
                        result_df.head()
                    )

                    # Download CSV
                    csv = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Processed CSV",
                        data=csv,
                        file_name="processed_reviews.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Failed to read CSV: {e}"
            )