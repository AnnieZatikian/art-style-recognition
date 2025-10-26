import pandas as pd

def load_artists_data(filepath):
    """
    Load artists.csv file.
    """
    return pd.read_csv(filepath)

def load_art_styles_data(filepath):
    """
    Load art_style.csv file.
    """
    return pd.read_csv(filepath)
