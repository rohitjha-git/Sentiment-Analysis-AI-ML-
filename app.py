import streamlit as st
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sentiment label mapping
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
st.title("📝 Sentiment Analysis Web App")
st.subheader("Analyze the sentiment of product reviews")

# User input
user_review = st.text_area("Enter a review:", "")

# Predict sentiment when the user clicks the button
if st.button("Analyze Sentiment"):
    if user_review.strip():
        # Transform the input text
        review_vectorized = vectorizer.transform([user_review])
        
        # Predict sentiment
        prediction = model.predict(review_vectorized)[0]
        sentiment = label_mapping[prediction]

        # Display the result
        st.write("### Sentiment: ", sentiment)
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
