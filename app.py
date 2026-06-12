import streamlit as st
import numpy as np

# --- Page Setup ---
st.set_page_config(
    page_title="BERT Simple Explainer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 BERT Sentiment & Attention Explainer")
st.write("See how BERT looks at an entire sentence simultaneously and uses Self-Attention to determine sentiment.")
st.divider()

# --- Simulating BERT Attention & Coding Engine ---
def run_simple_bert(text):
    # BERT adds a special [CLS] token at the front to capture global sentence sentiment
    raw_tokens = [t.strip().lower().strip(".,!?\"'") for t in text.split() if t.strip()]
    if not raw_tokens:
        return 0.5, [], None
        
    tokens = ["[CLS]"] + raw_tokens
    num_tokens = len(tokens)
    
    # Establish base sentiment scores for key words
    sentiment_weights = {
        "fantastic": 0.9, "incredible": 0.9, "great": 0.7, "love": 0.8, "good": 0.4,
        "bad": -0.5, "terrible": -0.8, "worst": -0.9, "boring": -0.6, "waste": -0.8
    }
    
    # Generate an Attention Matrix (how much each word looks at every other word)
    # Rows represent the source word, columns represent what it is paying attention to
    np.random.seed(sum(ord(c) for c in text) % 100)
    attention_matrix = np.random.uniform(0.05, 0.2, (num_tokens, num_tokens))
    
    # Make sure relevant words draw stronger attention to themselves
    for j, token in enumerate(tokens):
        if token in sentiment_weights:
            # Boost the column attention score for highly impactful words
            attention_matrix[:, j] += 0.4
            
    # Normalize rows so attention percentages sum up to 1.0 per word
    row_sums = attention_matrix.sum(axis=1, keepdims=True)
    attention_matrix = attention_matrix / row_sums
    
    # Calculate CLS vector embedding based on attention weights * sentiment scores
    cls_sentiment_signal = 0.0
    for j, token in enumerate(tokens):
        weight_from_cls = attention_matrix[0, j] # First row is the [CLS] token's gaze
        word_val = sentiment_weights.get(token, 0.0)
        cls_sentiment_signal += weight_from_cls * word_val
        
    final_score = 1 / (1 + np.exp(-cls_sentiment_signal * 5))
    return final_score, tokens, attention_matrix

# --- Layout Split ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Enter Your Text Input")
    user_input = st.text_input(
        "Type a sentence for BERT processing:",
        value="The acting was incredible, but the script was terrible."
    )
    
    st.markdown("""
    ### Key Transformer Concepts:
    * **`[CLS]` Token:** A dummy token inserted at position 0. Its final calculated state is what gets passed to a classification layer.
    * **Self-Attention:** A mathematical weighing process where words identify relevant structural relationships (e.g., matching the word *"acting"* with *"incredible"*).
    """)

with col_right:
    st.subheader("2. Transformer Prediction")
    
    if user_input.strip():
        score, tokens, att_matrix = run_simple_bert(user_input)
        
        if score >= 0.55:
            st.success(f"🟢 **BERT Predicts: POSITIVE** (Confidence: {score:.1%})")
        elif score <= 0.45:
            st.error(f"🔴 **BERT Predicts: NEGATIVE** (Confidence: {(1 - score):.1%})")
        else:
            st.warning(f"🟡 **BERT Predicts: NEUTRAL** (Confidence: {score:.1%})")
    else:
        st.info("Please type a phrase to observe attention weights.")

# --- Attention Matrix Breakdown ---
if user_input.strip() and tokens:
    st.divider()
    st.subheader("3. Interactive Self-Attention Matrix Mapping")
    st.write("Click on any token below to see how heavily the rest of the sentence relied on it during context processing:")
    
    # Display the attention weights assigned by the [CLS] token to other words
    cls_attention_weights = att_matrix[0]
    
    cols = st.columns(len(tokens))
    for idx, token in enumerate(tokens):
        with cols[idx]:
            # Highlight attention levels cleanly
            weight_pct = cls_attention_weights[idx]
            st.metric(label=f"'{token}'", value=f"{weight_pct:.1%}")
            if idx == 0:
                st.caption("Aggregator Token")
            elif token in ["incredible", "fantastic", "great", "love", "good", "bad", "terrible", "worst", "boring", "waste"]:
                st.caption("🔥 High Focus")
            else:
                st.caption("Context Buffer")

# --- Quick Math Section ---
st.divider()
st.subheader("📐 The Core Transformer Formula")
st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
st.caption("Instead of relying on recurrences to preserve distant words, BERT matches Queries ($Q$) and Keys ($K$) to establish alignment values, which map directly onto structural word Values ($V$).")