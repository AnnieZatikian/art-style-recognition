# Art Style Classifier

An interactive Streamlit-based web application for exploring artists and art styles — with machine learning integration to predict the style of a painting.

---

## Features

- **Artist Explorer**: Browse a curated list of artists with bios, images, and Wikipedia links.
- **Art Style Viewer**: Upload and view `art_style.csv` with built-in search.
- **Dark Themed UI**: Custom-styled interface with centered layout and landing image.
- **CSV Integration**: Loads and displays structured data (`artists.csv`, `art_style.csv`).
- **Model-Ready**: Supports TensorFlow `.h5` models for image-based classification (optional).
- **GitHub-Friendly**: Handles `.gitignore` for large files and clean commits.

---

## Project Structure

```
art_style_classifier/
├── app.py                  # Main Streamlit app
├── data/
│   ├── artists.csv         # Artist information (name, style, bio, link)
│   ├── art_style.csv       # Art style descriptions
│   ├── front_image.png     # Welcome image
├── models/                 # (Ignored from Git) Trained ML models
├── src/                    # Backend ML training and preprocessing code
├── .gitignore              # Git ignore rules
└── requirements.txt        # Python dependencies
```

---

## Getting Started

1. **Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/art_style_classifier.git
cd art_style_classifier
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

---

## Data Files

- `artists.csv`: Should include columns like `name`, `style`, `wikipedia`, `bio`
- `art_style.csv`: Contains `style`, `description`, `use` (used in the CSV viewer)
- `front_image.png`: Displayed on the landing screen

---


