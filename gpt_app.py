import streamlit as st
import numpy as np

# --- Page Setup ---
st.set_page_config(
    page_title="GPT Simple Explainer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GPT Causal Attention & Generation Explainer")
st.write("See how GPT processes text autoregressively from left to right, hiding future tokens using Causal Masking.")
st.divider()

# --- Simulating GPT Causal Attention Engine ---
def run_simple_gpt(text):
    tokens = [t.strip().lower().strip(".,!?\"'") for t in text.split() if t.strip()]
    if not tokens:
        return 0.5, [], None, ""
        
    num_tokens = len(tokens)
    
    sentiment_weights = {
        "fantastic": 0.8, "incredible": 0.8, "great": 0.6, "love": 0.7, "good": 0.4,
        "bad": -0.4, "terrible": -0.7, "worst": -0.8, "boring": -0.5, "waste": -0.7
    }
    
    # Initialize a strict Causal Attention Matrix (strictly lower triangular)
    np.random.seed(sum(ord(c) for c in text) % 100)
    attention_matrix = np.zeros((num_tokens, num_tokens))
    
    for i in range(num_tokens):
        for j in range(num_tokens):
            if j <= i:
                # Base random attention weight for past/current words
                weight = np.random.uniform(0.1, 0.3)
                # Boost attention if the historical word has strong sentiment properties
                if tokens[j] in sentiment_weights:
                    weight += 0.5
                attention_matrix[i, j] = weight
            else:
                # CAUSAL MASKING: Future words are forced to exactly 0.0 attention
                attention_matrix[i, j] = 0.0
                
        # Normalize the visible row tokens so valid weights sum to 1.0
        row_sum = attention_matrix[i].sum()
        if row_sum > 0:
            attention_matrix[i] = attention_matrix[i] / row_sum

    # The final token aggregates all cumulative context across the causal pipeline
    final_token_attention = attention_matrix[-1]
    accumulated_signal = 0.0
    for j, token in enumerate(tokens):
        accumulated_signal += final_token_attention[j] * sentiment_weights.get(token, 0.0)
        
    final_score = 1 / (1 + np.exp(-accumulated_signal * 4))
    
    # Simulate a "Next Token Prediction" generator response
    next_token = "👍" if final_score >= 0.55 else "👎" if final_score <= 0.45 else "💬"
    
    return final_score, tokens, attention_matrix, next_token

# --- Layout Split ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Enter Your Prompt")
    user_input = st.text_input(
        "Type a review sequence prefix for GPT:",
        value="The cinematic visuals were great but the dialogue was boring"
    )
    
    st.markdown("""
    ### Key GPT Concepts:
    * **Causal Masking:** A strict structural rule preventing time travel. Word #3 can look at words #1 and #2, but is blind to word #4.
    * **Next-Token Prediction:** Instead of outputting fixed class labels, GPT reads a prompt and calculates probability vectors to determine the next immediate character or word token.
    """)

with col_right:
    st.subheader("2. Autoregressive Output")
    
    if user_input.strip():
        score, tokens, att_matrix, predicted_emoji = run_simple_gpt(user_input)
        
        # Display Sentiment Inference derived from final sequence matrix
        if score >= 0.55:
            st.success(f"🟢 **Computed Trend: POSITIVE** (Confidence: {score:.1%})")
        elif score <= 0.45:
            st.error(f"🔴 **Computed Trend: NEGATIVE** (Confidence: {(1 - score):.1%})")
        else:
            st.warning(f"🟡 **Computed Trend: NEUTRAL** (Confidence: {score:.1%})")
            
        # Display Generation Simulation
        st.msb = st.code(f"Prompt: {user_input} ... [Predicted Next Token: {predicted_emoji}]")
    else:
        st.info("Please enter a text string sequence.")

# --- Left-to-Right Masked Attention Tracking ---
if user_input.strip() and tokens:
    st.divider()
    st.subheader("3. Causal Attention Matrix Map")
    st.write("See how each row is physically restricted to only attending to **past and present** tokens:")

    for i, target_word in enumerate(tokens):
        with st.expander(f"Word position {i+1}: '{target_word}' looks back at..."):
            
            # Create dynamic visibility columns matching the current token limits
            cols = st.columns(len(tokens))
            for j, source_word in enumerate(tokens):
                with cols[j]:
                    attention_weight = att_matrix[i, j]
                    
                    if j < i:
                        st.metric(label=f"'{source_word}'", value=f"{attention_weight:.0%}")
                        st.caption("👈 Past Context")
                    elif j == i:
                        st.metric(label=f"'{source_word}'", value=f"{attention_weight:.0%}")
                        st.caption("🎯 Current Word")
                    else:
                        # Masked future positions
                        st.metric(label=f"'{source_word}'", value="🚫")
                        st.caption("🔒 Masked (Future)")

# --- Quick Math Section ---
st.divider()
st.subheader("📐 The Causal Masking Mechanism")
st.latex(r"\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V")
st.caption("To physically block future data leakages, GPT adds a Mask matrix ($M$) where all future positions are set to $-\infty$. This forces the softmax score to drop to exactly $0$, rendering future tokens completely invisible.")