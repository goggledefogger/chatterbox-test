# Chatterbox Studio

**Chatterbox Studio** is a Streamlit interface for open-source Text-to-Speech models.

This project is powered by the **[Chatterbox](https://github.com/resemble-ai/chatterbox)** library by **[Resemble AI](https://resemble.ai)**.

It provides a workspace to test and integrate the Chatterbox family of models (Turbo, English, and Multilingual) with support for paralinguistic tags and voice cloning.

## Key Features

- **Chatterbox-Turbo Support**: Use paralinguistic tags like `[chuckle]`, `[sigh]`, and `[cough]`.
- **Progress Tracking**: Functional feedback during model loading and audio synthesis directly in the UI.
- **Hugging Face Integration**: Token input with real-time download progress.
- **Voice Cloning**: Zero-shot cloning using WAV/MP3 reference audio.
- **Apple Silicon Support**: Uses Metal Performance Shaders (MPS) for inference on Mac.

---

## Development Setup

We recommend using [uv](https://github.com/astral-sh/uv) for package management.

### 1. Clone the repository
```bash
git clone https://github.com/goggledefogger/chatterbox-studio.git
cd chatterbox-studio
```

### 2. Create and activate a Virtual Environment
```bash
uv venv --python 3.11.9
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -e .
```

### 4. Run the Studio
```bash
streamlit run streamlit_app.py
```

---

## Development Guidelines

- **Environment**: Tested on Python 3.11.
- **Model Loading**: Models are cached using `@st.cache_resource` to manage memory.
- **Architecture**: Uses `StreamlitTqdm` to monkeypatch `tqdm` for capturing library status updates.

---

## 🙏 Credits

This project uses work by:
- **[Resemble AI](https://resemble.ai)** for the **[Chatterbox](https://github.com/resemble-ai/chatterbox)** models and library.
- **[Streamlit](https://streamlit.io)** for the application framework.
- **[Hugging Face](https://huggingface.co)** for hosting the model weights and `transformers` ecosystem.

---

## ⚖️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.

---

Built with ❤️ by [Danny](https://github.com/goggledefogger)
