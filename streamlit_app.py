import streamlit as st
import tqdm
import threading

# Page configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="Chatterbox Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

class StreamlitTqdm:
    """A simple tqdm-like wrapper for Streamlit progress bars."""
    _lock = threading.Lock()

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def __init__(self, iterable=None, desc=None, total=None, leave=True, file=None, unit="B", *args, **kwargs):
        self.iterable = iterable
        self.desc = desc or "Processing"
        self.total = total
        self.unit = unit
        self.n = 0
        # Only create UI elements once
        self.container = st.empty()
        self.pbar = self.container.progress(0, text=f"{self.desc}: 0%")

    def __iter__(self):
        if self.iterable is not None:
            for obj in self.iterable:
                yield obj
                self.update(1)

    def update(self, n=1):
        self.n += n
        if self.total:
            progress = min(max(self.n / self.total, 0.0), 1.0)
            percentage = int(progress * 100)
            self.pbar.progress(progress, text=f"{self.desc}: {percentage}%")
        else:
            self.pbar.progress(0, text=f"{self.desc}: {self.n} {self.unit}")

    def close(self):
        # We leave it visible so the user sees 100% at the end
        pass

    def set_description(self, desc, refresh=True):
        self.desc = desc
        self.update(0)

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

    def moveto(self, n):
        pass

    def clear(self, *args, **kwargs):
        pass

    def display(self, *args, **kwargs):
        pass

    def write(self, s, *args, **kwargs):
        st.write(s)

    def reset(self, total=None):
        if total:
            self.total = total
        self.n = 0
        self.update(0)

    def __getattr__(self, name):
        """Catch-all for any other tqdm methods to prevent crashes."""
        def dummy(*args, **kwargs):
            return self
        return dummy

# Monkeypatch tqdm BEFORE any other imports that might use it
tqdm.tqdm = StreamlitTqdm

import torch
import torchaudio as ta
import numpy as np
import tempfile
import os
import time
import gc
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3 import T3
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import EnTokenizer, MTLTokenizer
from chatterbox.models.t3.modules.t3_config import T3Config
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container

# Device detection
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stButton>button {
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .tag-button {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 13px;
        cursor: pointer;
        margin: 2px;
        display: inline-block;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .sub-header {
        color: #6B7280;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "current_audio" not in st.session_state:
    st.session_state.current_audio = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Chatterbox-Turbo"

# --- Models ---
MODELS = {
    "Chatterbox-Turbo": ChatterboxTurboTTS,
    "Chatterbox (English)": ChatterboxTTS,
    "Chatterbox (Multilingual)": ChatterboxMultilingualTTS
}

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

@st.cache_resource(max_entries=1)
def load_model(name, device):
    # Clear existing memory before loading a new model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    with st.status(f"Initializing {name}...", expanded=True) as status:
        # Use None if no token is provided to avoid LocalTokenNotFoundError
        token = os.getenv("HF_TOKEN") or None
        map_location = torch.device('cpu') if device in ["cpu", "mps"] else None

        try:
            if name == "Chatterbox-Turbo":
                repo_id = "ResembleAI/chatterbox-turbo"
                st.write(f"📥 Checking `{repo_id}`...")
                ckpt_dir = Path(snapshot_download(
                    repo_id=repo_id,
                    token=token,
                    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
                    tqdm_class=StreamlitTqdm
                ))

                st.write("📻 Loading Voice Encoder...")
                ve = VoiceEncoder()
                ve.load_state_dict(load_safetensors(ckpt_dir / "ve.safetensors"))
                ve.to(device, dtype=DTYPE).eval()

                st.write("🧠 Initializing T3 Transformer (Turbo)...")
                hp = T3Config(text_tokens_dict_size=50276)
                hp.llama_config_name = "GPT2_medium"
                hp.speech_tokens_dict_size = 6563
                hp.input_pos_emb = None
                hp.speech_cond_prompt_len = 375
                hp.use_perceiver_resampler = False
                hp.emotion_adv = False

                t3 = T3(hp)
                t3_state = load_safetensors(ckpt_dir / "t3_turbo_v1.safetensors")
                if "model" in t3_state.keys():
                    t3_state = t3_state["model"][0]
                t3.load_state_dict(t3_state)
                if hasattr(t3.tfmr, 'wte'):
                    del t3.tfmr.wte

                st.write("Moving T3 weights to GPU...")
                t3.to(device, dtype=DTYPE).eval()

                st.write("🎙️ Preparing S3 Generator...")
                s3gen = S3Gen(meanflow=True)
                s3gen_state = load_safetensors(ckpt_dir / "s3gen_meanflow.safetensors")
                s3gen.load_state_dict(s3gen_state, strict=True)
                s3gen.to(device, dtype=DTYPE).eval()

                # Explicitly clear state dicts and collect garbage
                del t3_state
                del s3gen_state
                gc.collect()

                st.write("📖 Loading Tokenizer...")
                # Turbo uses AutoTokenizer from transformers
                tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                conds = None
                if (ckpt_dir / "conds.pt").exists():
                    from chatterbox.tts_turbo import Conditionals
                    conds = Conditionals.load(ckpt_dir / "conds.pt", map_location=map_location).to(device, dtype=DTYPE)

                model = ChatterboxTurboTTS(t3, s3gen, ve, tokenizer, device, conds=conds)

            elif name == "Chatterbox (English)":
                repo_id = "ResembleAI/chatterbox"
                st.write(f"📥 Checking `{repo_id}`...")
                files = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
                local_f = None
                for f in files:
                    local_f = hf_hub_download(repo_id=repo_id, filename=f, token=token, tqdm_class=StreamlitTqdm)
                ckpt_dir = Path(os.path.dirname(local_f))

                st.write("📻 Loading Voice Encoder...")
                ve = VoiceEncoder()
                ve.load_state_dict(load_safetensors(ckpt_dir / "ve.safetensors"))
                ve.to(device, dtype=DTYPE).eval()

                st.write("🧠 Initializing T3 Transformer (English)...")
                t3 = T3()
                t3_state = load_safetensors(ckpt_dir / "t3_cfg.safetensors")
                if "model" in t3_state.keys():
                    t3_state = t3_state["model"][0]
                t3.load_state_dict(t3_state)

                st.write("⚡ Moving T3 weights to GPU...")
                t3.to(device, dtype=DTYPE).eval()

                st.write("🎙️ Preparing S3 Generator...")
                s3gen = S3Gen()
                s3gen_state = load_safetensors(ckpt_dir / "s3gen.safetensors")
                s3gen.load_state_dict(s3gen_state, strict=False)
                s3gen.to(device, dtype=DTYPE).eval()

                # Explicitly clear state dicts and collect garbage
                del t3_state
                del s3gen_state
                gc.collect()

                st.write("📖 Loading Tokenizer...")
                tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

                conds = None
                if (ckpt_dir / "conds.pt").exists():
                    from chatterbox.tts import Conditionals
                    conds = Conditionals.load(ckpt_dir / "conds.pt", map_location=map_location).to(device, dtype=DTYPE)

                model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)

            elif name == "Chatterbox (Multilingual)":
                repo_id = "ResembleAI/chatterbox"
                st.write(f"📥 Checking `{repo_id}`...")
                ckpt_dir = Path(snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision="main",
                    allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                    token=token,
                    tqdm_class=StreamlitTqdm
                ))

                st.write("📻 Loading Voice Encoder...")
                ve = VoiceEncoder()
                ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location='cpu'))
                ve.to(device, dtype=DTYPE).eval()

                st.write("🧠 Initializing T3 Transformer (Multilingual)...")
                t3 = T3(T3Config.multilingual())
                t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
                if "model" in t3_state.keys():
                    t3_state = t3_state["model"][0]
                t3.load_state_dict(t3_state)

                st.write("⚡ Moving T3 weights to GPU...")
                t3.to(device, dtype=DTYPE).eval()

                st.write("🎙️ Preparing S3 Generator...")
                s3gen = S3Gen()
                s3gen_state = torch.load(ckpt_dir / "s3gen.pt", weights_only=True, map_location='cpu')
                s3gen.load_state_dict(s3gen_state)
                s3gen.to(device, dtype=DTYPE).eval()

                # Explicitly clear state dicts and collect garbage
                del t3_state
                del s3gen_state
                gc.collect()

                st.write("📖 Loading MTL Tokenizer...")
                tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

                conds = None
                if (ckpt_dir / "conds.pt").exists():
                    from chatterbox.mtl_tts import Conditionals
                    conds = Conditionals.load(ckpt_dir / "conds.pt", map_location=map_location).to(device, dtype=DTYPE)

                model = ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)

            status.update(label=f"✅ {name} Ready!", state="complete", expanded=False)
            return model

        except Exception as e:
            status.update(label="❌ Load Failed", state="error", expanded=True)
            st.error(f"Failed to load {name}: {str(e)}")
            if "Token is required" in str(e):
                st.warning("⚠️ This model might required a Hugging Face token. Please provide one in the sidebar.")
            raise e

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Model Selection")
    model_name = st.selectbox("Choose Model", list(MODELS.keys()), index=0)

    st.markdown("---")
    st.markdown("### Hugging Face Integration")
    hf_token = st.text_input("HF Token", type="password", help="Enter your Hugging Face token to access gated models.")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        st.success("HF Token set!")

    st.markdown("---")
    st.markdown("### Generation Parameters")
    temp = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05)
    top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.01)
    top_k = st.slider("Top K", 0, 1000, 1000, 10)
    rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.05)

    if model_name == "Chatterbox-Turbo":
        norm_loudness = st.checkbox("Normalize Loudness", value=True)
    else:
        exaggeration = st.slider("Exaggeration", 0.0, 1.0, 0.5, 0.1)
        cfg_weight = st.slider("CFG Weight", 0.0, 1.0, 0.5, 0.1)

    st.markdown("---")
    st.markdown("### Memory Management")
    if st.button("🧼 Purge Memory", help="Manually clear model cache and hardware VRAM."):
        st.cache_resource.clear()
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
        st.success("Memory purged!")
        st.rerun()

# --- Main App ---
st.markdown('<p class="main-header">Chatterbox Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Open Source TTS by Resemble AI</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Synthesis")

    default_text = "Oh, that's hilarious! [chuckle] Anyway, we have a new model. It's built for speed and quality." if model_name == "Chatterbox-Turbo" else "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus."

    input_text = st.text_area("Text to Synthesize", value=default_text, height=150, help="Max recommendation 300 characters for best performance.")

    # Paralinguistic Tags (only for Turbo)
    if model_name == "Chatterbox-Turbo":
        st.markdown("#### Paralinguistic Tags")
        tag_cols = st.columns(len(EVENT_TAGS) // 3 + 1)
        for i, tag in enumerate(EVENT_TAGS):
            with tag_cols[i // 3]:
                if st.button(tag, use_container_width=True):
                    # Note: injection is hard in pure streamlit, but we can show instructions
                    st.info(f"Add `{tag}` to your text to hear it!")

    st.markdown("---")
    st.markdown("### Voice Cloning")
    ref_audio = st.file_uploader("Upload reference audio (WAV/MP3)", type=["wav", "mp3"])
    st.info("💡 If no reference is provided, the model will use its default voice.")

with col2:
    st.markdown("### Output")
    output_area = st.empty()
    if st.button("Generate Audio", type="primary", use_container_width=True):
        st.session_state.current_audio = None
        if not input_text:
            st.error("Please enter some text!")
        else:
            # We use a status block for synthesis as well
            with st.status("Synthesizing...", expanded=True) as status:
                st.write("🧠 Loading model...")
                model = load_model(model_name, DEVICE)

                start_time = time.time()
                st.write("🎙️ Preparing conditionals and text...")

                # Setup kwargs
                kwargs = {
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": int(top_k),
                    "repetition_penalty": rep_penalty,
                }

                if model_name == "Chatterbox-Turbo":
                    kwargs["norm_loudness"] = norm_loudness
                else:
                    kwargs["exaggeration"] = exaggeration
                    kwargs["cfg_weight"] = cfg_weight

                # Handle reference audio
                temp_ref_path = None
                if ref_audio:
                    st.write("📎 Processing reference audio...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(ref_audio.read())
                        temp_ref_path = tmp.name
                        kwargs["audio_prompt_path"] = temp_ref_path

                try:
                    # Select language for Multilingual
                    if model_name == "Chatterbox (Multilingual)":
                        kwargs["language_id"] = "en" # Default to English for demo

                    st.write("Generating tokens and waveform...")
                    wav = model.generate(input_text, **kwargs)

                    if temp_ref_path:
                        os.unlink(temp_ref_path)

                    end_time = time.time()
                    status.update(label=f"Completed in {end_time - start_time:.2f}s", state="complete", expanded=False)

                    # Store in session state
                    st.session_state.current_audio = (model.sr, wav.squeeze(0).cpu().numpy())

                except Exception as e:
                    status.update(label="❌ Error", state="error", expanded=True)
                    st.error(f"Error during synthesis: {e}")

    if st.session_state.current_audio:
        sr, audio_np = st.session_state.current_audio
        st.audio(audio_np, sample_rate=sr)

        # Download button
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            ta.save(tmp_out.name, torch.from_numpy(audio_np).unsqueeze(0), sr)
            with open(tmp_out.name, "rb") as f:
                st.download_button("Download WAV", f, file_name="chatterbox_output.wav", mime="audio/wav")
            os.unlink(tmp_out.name)

add_vertical_space(5)
st.markdown("---")
st.markdown("Using [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI.")
