from src.data_loader import load_artists_data, load_art_styles_data

# Load the CSV files
artists_df = load_artists_data('data/artists.csv')
styles_df = load_art_styles_data('data/art_style.csv')

# Print some info
print("Artists Data:")
print(artists_df.head())

print("\nArt Styles Data:")
print(styles_df.head())
