import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

def perform_topic_modeling():
    print("--- Initiating Topic Modeling (LDA) ---")
    # Mock documents (e.g., product reviews)
    documents = [
        "The battery life on this laptop is amazing and lasts all day.",
        "Customer service was terrible and I waited on hold for hours.",
        "Great screen resolution, very vibrant colors.",
        "The battery drains too fast when playing games.",
        "Support team helped me fix my billing issue quickly."
    ]

    # Vectorization
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(documents)

    # LDA Model (Extracting 2 main topics)
    lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
    lda_model.fit(doc_term_matrix)

    # Displaying topics
    feature_names = vectorizer.get_feature_names_out()
    for index, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-3:]]
        print(f"Topic #{index + 1}: {', '.join(top_words)}")
    print("\n")

def perform_summarization():
    print("--- Initiating Abstractive Summarization ---")
    long_text = (
        "Artificial intelligence is rapidly changing the world. Companies are adopting machine "
        "learning models to optimize their operations and improve customer experiences. Despite the "
        "excitement, there are concerns about job displacement and data privacy. However, experts "
        "believe that AI will create new types of jobs and that strong regulations can mitigate "
        "privacy risks, ultimately leading to a more efficient society."
    )
    print(f"Original Text Length: {len(long_text)} characters")

    try:
        # Using HuggingFace Transformers for Abstractive Summarization
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        summary = summarizer(long_text, max_length=40, min_length=15, do_sample=False)
        print(f"Generated Summary: {summary[0]['summary_text']}")
    except Exception as e:
        print("Note: Install PyTorch and Transformers to run the summarization model locally.")

if __name__ == "__main__":
    perform_topic_modeling()
    perform_summarization()