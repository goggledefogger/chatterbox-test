# ⚡ Chatterbox Studio

**Chatterbox Studio** is a high-performance, developer-friendly Streamlit interface for state-of-the-art open-source Text-to-Speech models.

This project is maintained by **[Danny (goggledefogger)](https://github.com/goggledefogger)** and is powered by the **[Chatterbox](https://github.com/resemble-ai/chatterbox)** library by **[Resemble AI](https://resemble.ai)**.

It provides an intuitive workspace to test, tweak, and integrate the Chatterbox family of models (Turbo, English, and Multilingual) with native support for paralinguistic tags and voice cloning.

## 🚀 Key Features

- **Chatterbox-Turbo Native Support**: Effortlessly use paralinguistic tags like `[chuckle]`, `[sigh]`, and `[cough]`.
- **Granular Progress Tracking**: Real-time feedback during model loading and audio synthesis directly in the UI.
- **Hugging Face Integration**: Secure token input with real-time download progress.
- **Voice Cloning**: Zero-shot cloning using standard WAV/MP3 reference audio.
- **M1/M2/M3 Optimized**: Leveraging Metal Performance Shaders (MPS) for lightning-fast inference on Apple Silicon.

---

## 🛠️ Development Setup

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

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

## 📖 Development Guidelines

- **Environment**: Developed and tested on Python 3.11.
- **Model Loading**: Models are cached using `@st.cache_resource` to prevent redundant memory allocation.
- **Architecture**: The app uses a robust, thread-safe progress tracking system (`StreamlitTqdm`) that monkeypatches `tqdm` to capture all backend library status updates.

---

## 🙏 Credits

This project would not be possible without the incredible open-source work by:
- **[Resemble AI](https://resemble.ai)** for the **[Chatterbox](https://github.com/resemble-ai/chatterbox)** models and library.
- **[Hugging Face](https://huggingface.co)** for hosting the model weights and `transformers` ecosystem.

---

## ⚖️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.

---

Built with ❤️ by [Danny](https://github.com/goggledefogger)
