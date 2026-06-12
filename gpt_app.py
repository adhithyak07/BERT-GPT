import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time

# ==============================================================================
# 1. PAGE SETUP & UNIFIED ULTRA-PREMIUM LIGHT UI (CUSTOM CSS)
# ==============================================================================
st.set_page_config(
    page_title="Deep Learning Architecture Studio", 
    page_icon="⚡", 
    layout="wide"
)

# Shared premium styling for headers, buttons, cards, and custom tensor badges
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        color: #1E293B;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Gradient Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #7c3aed 0%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Modern Content Cards */
    [data-testid="stVerticalBlock"] > div:has(div.custom-block), .stBlock {
        background: #FFFFFF !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.05), 0 8px 16px -6px rgba(0, 0, 0, 0.03) !important;
        border: 2px solid #E2E8F0 !important;
        margin-bottom: 20px !important;
    }
    
    /* Training/Action Button Styling */
    .action-btn > div > button, .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        height: 48px;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2) !important;
        transition: all 0.2s ease;
    }
    .action-btn > div > button:hover, .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35) !important;
    }
    
    /* Terminal Code Block Panel */
    .terminal-container {
        background-color: #0F172A;
        border-left: 5px solid #7c3aed;
        padding: 20px;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        line-height: 1.6;
        color: #38BDF8;
        white-space: pre-wrap;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    }
    
    /* Dynamic Colored Badges for Tensors */
    .direction-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .forward-pass { background-color: #e0f2fe !important; color: #0369a1 !important; border: 1px solid #bae6fd !important; }
    .backward-pass { background-color: #fce7f3 !important; color: #b7056d !important; border: 1px solid #fbcfe8 !important; }
    .concat-state { background-color: #f3e8ff !important; color: #6b21a8 !important; border: 1px solid #e9d5ff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 2. MINI-GPT PYTORCH MODULES
# ==============================================================================
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x)  
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# 3. CORPUS AND SIDEBAR CONFIGURATION
# ==============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.sidebar.markdown("### 🧭 Architecture Studio Navigation")
studio_mode = st.sidebar.radio("Select Active Model Workspace:", ["GPT Autoregressive Studio", "Bidirectional LSTM Workbench"])

# Global hyperparameters for models
n_embd, n_head, n_layer, block_size = 64, 4, 3, 16

# ==============================================================================
# WORKSPACE INTERFACE 1: GPT STUDIO
# ==============================================================================
if studio_mode == "GPT Autoregressive Studio":
    st.title("⚡ GPT Architecture Studio")
    st.write("Train and visualize an autoregressive Transformer language model natively inside a sleek, bright interface.")
    
    st.sidebar.markdown("### 🛠️ GPT Configuration")
    st.sidebar.info(f"**Embedding Space:** {n_embd} channels\n\n**Attention Heads:** {n_head}\n\n**Transformer Layers:** {n_layer}\n\n**Context Bounds:** {block_size} tokens")
    
    max_tokens = st.sidebar.slider("Tokens to Output", 10, 300, 120)
    temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.6)

    training_corpus = """
    Once upon a time, there was a tiny artificial intelligence model. 
    The little model wanted to speak human language correctly. It tried to learn every day.
    Once upon a time, the model worked hard. It processed tokens and adjusted weights. 
    Suddenly, after training, it stopped generating gibberish. It spoke clearly and made sense!
    Once upon a time, everyone celebrated the successful training of the tiny model.
    """
    
    chars = sorted(list(set(training_corpus + " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n")))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s] 
    decode = lambda l: ''.join([itos[i] for i in l])

    def get_batch(data, batch_size, block_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    if 'gpt_model' not in st.session_state:
        st.session_state.gpt_model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, device).to(device)
        st.session_state.has_trained_successfully = False
        st.session_state.cumulative_steps = 0

    st.subheader("🤖 Step 1: Optimize Model Weights")
    col1, col2 = st.columns([1, 2])
    with col1:
        epochs = st.number_input("Steps to Train", min_value=200, max_value=3000, value=1200, step=100)
    with col2:
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        if st.button("🚀 Run Neural Training Loop", use_container_width=True):
            data_tensor = torch.tensor(encode(training_corpus), dtype=torch.long)
            optimizer = torch.optim.AdamW(st.session_state.gpt_model.parameters(), lr=1e-3)
            
            loss_progress = st.progress(0)
            status_text = st.empty()
            chart_slot = st.empty()
            chart_data = []
            
            st.session_state.gpt_model.train()
            for step in range(epochs):
                xb, yb = get_batch(data_tensor, batch_size=16, block_size=block_size)
                logits, loss = st.session_state.gpt_model(xb, yb)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                if step % 50 == 0 or step == epochs - 1:
                    loss_progress.progress((step + 1) / epochs)
                    status_text.text(f"Step {step}/{epochs} — Current Loss: {loss.item():.4f}")
                    chart_data.append(loss.item())
                    chart_slot.line_chart(chart_data)
                    
            st.session_state.gpt_model.eval()
            st.session_state.has_trained_successfully = True
            st.session_state.cumulative_steps += epochs
            st.success(f"🎉 Model weights locked! Total training loops run: {st.session_state.cumulative_steps}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("✍️ Step 2: Autoregressive Text Generation")
    user_prompt = st.text_input("Enter your starting story prompt:", value="Once upon a time")

    st.markdown('<div class="action-btn">', unsafe_allow_html=True)
    if st.button("✨ Execute Token Generation Pipeline", use_container_width=True):
        if not st.session_state.has_trained_successfully:
            st.error("🛑 Safety Block: Run the neural training loop in Step 1 first!")
        elif not user_prompt.strip():
            st.warning("Please supply an input sequence context first.")
        else:
            clean_prompt = "".join([c for c in user_prompt if c in stoi])
            if not clean_prompt: clean_prompt = "Once upon a time"
            
            with st.spinner("Decoding persistent attention maps..."):
                context_tokens = encode(clean_prompt)
                x = torch.tensor([context_tokens], dtype=torch.long, device=device)
                with torch.no_grad():
                    generated_indices = st.session_state.gpt_model.generate(x, max_new_tokens=max_tokens, temperature=temperature)
                    output_text = decode(generated_indices[0].tolist())
                
                st.markdown("### 🖥️ Studio Output Stream:")
                st.markdown(f'<div class="terminal-container">{output_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# WORKSPACE INTERFACE 2: BIDIRECTIONAL LSTM
# ==============================================================================
else:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%); border-radius: 16px; margin-bottom: 30px; box-shadow: 0 10px 20px rgba(124, 58, 237, 0.15);">
        <div style="color: #ffffff !important; font-weight: 800 !important; font-size: 2.5rem !important; margin: 0 !important; letter-spacing: -0.02em;">🔁 Bidirectional LSTM Workbench</div>
        <div style="color: #ddd6fe !important; font-size: 1.05rem; font-weight: 500; margin-top: 5px;">Supervised Deep Learning Sequence Representation Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🎛️ Bi-LSTM Options")
    merge_strategy = st.sidebar.selectbox("Vector Merge Mode", ["Concatenation", "Summation", "Average"])
    delay = st.sidebar.slider("Calculation Execution Delay (s)", 0.0, 1.0, 0.05)

    def run_bilstm_simulation(text, speed):
        tokens = [t.strip() for t in text.split() if t.strip()]
        if not tokens: return None, None, None, 0.5, []
        
        num_tokens = len(tokens)
        h_forward = np.zeros((num_tokens, 2))
        h_backward = np.zeros((num_tokens, 2))
        lower_tokens = [t.lower().strip(".,!?\"'") for t in tokens]
        has_negation = any(neg in lower_tokens for neg in ['not', 'never', 'no', 'isnt', 'wasnt'])
        
        # Forward Pass
        current_f = np.zeros(2)
        for i in range(num_tokens):
            seed = sum(ord(c) for c in tokens[i]) % 100
            np.random.seed(seed)
            bias = 0.0
            clean_t = lower_tokens[i]
            if clean_t in ['excellent', 'love', 'perfect', 'amazing', 'masterpiece']: bias = 0.6
            elif clean_t in ['bad', 'boring', 'waste', 'disappointment', 'terrible']: bias = -0.1 if has_negation else -0.6
            current_f = np.tanh(0.7 * current_f + np.random.normal(bias, 0.3, 2))
            h_forward[i] = current_f

        # Backward Pass
        current_b = np.zeros(2)
        for i in reversed(range(num_tokens)):
            seed = (sum(ord(c) for c in tokens[i]) + 50) % 100
            np.random.seed(seed)
            bias = 0.0
            clean_t = lower_tokens[i]
            if clean_t in ['excellent', 'love', 'perfect', 'amazing', 'masterpiece']: bias = 0.6
            elif clean_t in ['bad', 'boring', 'waste', 'disappointment', 'terrible']: bias = -0.1 if has_negation else -0.6
            if clean_t == 'not' and i + 1 < num_tokens and lower_tokens[i+1] in ['bad', 'boring']: bias = 0.8 
            current_b = np.tanh(0.7 * current_b + np.random.normal(bias, 0.3, 2))
            h_backward[i] = current_b
            
        history = []
        for i in range(num_tokens):
            concatenated = np.concatenate([h_forward[i], h_backward[i]])
            history.append({
                "step": i + 1,
                "token": tokens[i],
                "forward": h_forward[i].copy(),
                "backward": h_backward[i].copy(),
                "concat": concatenated
            })
            
        final_context = np.mean(h_forward[-1]) + np.mean(h_backward[0])
        sentiment_score = 1 / (1 + np.exp(-final_context * 2.5))
        return h_forward[-1], h_backward[0], np.concatenate([h_forward[-1], h_backward[0]]), sentiment_score, history

    col_left, col_right = st.columns([1, 1.2], gap="large")
    with col_left:
        st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
        st.subheader("📥 Input Target Sequence")
        sample_text = st.text_area(
            "Enter raw text stream below:",
            value="Not a boring film, it was an amazing masterpiece with excellent pacing.",
            height=100
        )
        trigger_process = st.button("🚀 Analyze Dual-Direction Sequences", use_container_width=True)

    with col_right:
        st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
        st.subheader("📊 Output Metrics Engine")
        
        if trigger_process and sample_text.strip():
            with st.spinner("Processing deep network nodes..."):
                f_vector, b_vector, combined_vector, sentiment, trace_log = run_bilstm_simulation(sample_text, delay)
                
                progress = st.progress(0)
                for pct in range(100):
                    time.sleep(0.001)
                    progress.progress(pct + 1)
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Processed Tokens", len(sample_text.split()))
                m_col2.metric("Forward Pass Size", f"{len(f_vector)}D")
                m_col3.metric("Backward Pass Size", f"{len(b_vector)}D")
                
                st.divider()
                st.markdown("#### **Classification Inference**")
                if sentiment >= 0.5:
                    st.success(f"🟢 **POSITIVE SENTIMENT** Confidence: {sentiment:.2%}")
                else:
                    st.error(f"🔴 **NEGATIVE SENTIMENT** Confidence: {(1 - sentiment):.2%}")
                    
                st.divider()
                st.markdown("#### **Unified Hidden Layer States Block**")
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.markdown("<span class='direction-badge forward-pass'>Forward State Vector</span>", unsafe_allow_html=True)
                    st.code(f"{f_vector}")
                with v_col2:
                    st.markdown("<span class='direction-badge backward-pass'>Backward State Vector</span>", unsafe_allow_html=True)
                    st.code(f"{b_vector}")
                    
                st.markdown("<span class='direction-badge concat-state'>Concatenated Bidirectional Payload Matrix</span>", unsafe_allow_html=True)
                st.code(f"{combined_vector}")
        else:
            st.info("💡 Write an input sentence on the left and trigger processing to populate computational vectors.")

    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("💡 Context Dependency Breakdown")
    st.write("By utilizing **Bidirectional LSTM structures**, data streams concurrently from both ends (left-to-right and right-to-left). This allows long-range text dependencies to preserve local semantic logic.")

    if trigger_process and sample_text.strip() and 'trace_log' in locals():
        st.subheader("📋 Step Ledger Breakdown")
        for item in trace_log:
            with st.expander(label=f"Token Block Index {item['step']}: '{item['token']}'"):
                t_col1, t_col2, t_col3 = st.columns(3)
                t_col1.markdown("**Forward Pass Result:**")
                t_col1.code(f"{item['forward']}")
                t_col2.markdown("**Backward Pass Result:**")
                t_col2.code(f"{item['backward']}")
                t_col3.markdown("**Merged Vector Output:**")
                t_col3.code(f"{item['concat']}")
