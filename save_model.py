# save_model.py
# Downloads Book Crossing dataset and builds recommendation models

import os
import pickle
import pandas as pd
import numpy as np
import warnings
import requests
import zipfile
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

warnings.filterwarnings('ignore')


# ── Load Google Books API Key ─────────────────────────────────
GOOGLE_BOOKS_KEY = os.getenv('GOOGLE_BOOKS_KEY', '')


# ── Test Google Books API ─────────────────────────────────────
print("Testing Google Books API...")
if GOOGLE_BOOKS_KEY:
    try:
        url      = (
            f"https://www.googleapis.com/books/v1/volumes?"
            f"q=harry+potter&key={GOOGLE_BOOKS_KEY}"
        )
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✅ Google Books API working!")
        else:
            print(f"⚠️ Status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error: {e}")
else:
    print("⚠️ No Google Books API key found")


# ── Download Dataset ──────────────────────────────────────────
print("\nDownloading Book Crossing dataset...")
try:
    url      = "https://github.com/dhairavc/DATA612-RecommenderSystems/raw/master/Final%20Project/BX-CSV-Dump.zip"
    response = requests.get(url, timeout=120)
    zip_data = zipfile.ZipFile(io.BytesIO(response.content))
    zip_data.extractall('/tmp/bookcrossing/')
    print("✅ Dataset downloaded!")
    print("Files extracted:")
    for f in os.listdir('/tmp/bookcrossing/'):
        print(f"  {f}")
except Exception as e:
    print(f"Download failed: {e}")
    raise


# ── Load CSV Files ────────────────────────────────────────────
print("\nLoading CSV files...")
try:
    books = pd.read_csv(
        '/tmp/bookcrossing/BX-Books.csv',
        sep=';', encoding='latin-1',
        on_bad_lines='skip',
        low_memory=False
    )
    ratings = pd.read_csv(
        '/tmp/bookcrossing/BX-Book-Ratings.csv',
        sep=';', encoding='latin-1',
        on_bad_lines='skip'
    )
    users = pd.read_csv(
        '/tmp/bookcrossing/BX-Users.csv',
        sep=';', encoding='latin-1',
        on_bad_lines='skip'
    )
    print(f"Books   shape: {books.shape}")
    print(f"Ratings shape: {ratings.shape}")
    print(f"Users   shape: {users.shape}")
    print(f"\nBooks columns  : {books.columns.tolist()}")
    print(f"Ratings columns: {ratings.columns.tolist()}")
    print(f"Users columns  : {users.columns.tolist()}")
except Exception as e:
    print(f"Error loading files: {e}")
    raise


# ── Clean Column Names ────────────────────────────────────────
print("\nCleaning column names...")
books.columns   = [c.strip().strip('"') for c in books.columns]
ratings.columns = [c.strip().strip('"') for c in ratings.columns]
users.columns   = [c.strip().strip('"') for c in users.columns]

# Rename to standard names
books_col_map = {}
for col in books.columns:
    cl = col.lower().replace('-','_').replace(' ','_')
    if 'isbn' in cl:
        books_col_map[col] = 'ISBN'
    elif 'title' in cl:
        books_col_map[col] = 'title'
    elif 'author' in cl:
        books_col_map[col] = 'author'
    elif 'year' in cl:
        books_col_map[col] = 'year'
    elif 'publisher' in cl:
        books_col_map[col] = 'publisher'
    elif 'url_s' in cl or 'image_s' in cl:
        books_col_map[col] = 'image_s'
    elif 'url_m' in cl or 'image_m' in cl:
        books_col_map[col] = 'image_m'
    elif 'url_l' in cl or 'image_l' in cl:
        books_col_map[col] = 'image_l'
books.rename(columns=books_col_map, inplace=True)

ratings_col_map = {}
for col in ratings.columns:
    cl = col.lower().replace('-','_').replace(' ','_')
    if 'user' in cl:
        ratings_col_map[col] = 'user_id'
    elif 'isbn' in cl:
        ratings_col_map[col] = 'ISBN'
    elif 'rating' in cl:
        ratings_col_map[col] = 'rating'
ratings.rename(columns=ratings_col_map, inplace=True)

users_col_map = {}
for col in users.columns:
    cl = col.lower().replace('-','_').replace(' ','_')
    if 'user' in cl:
        users_col_map[col] = 'user_id'
    elif 'location' in cl:
        users_col_map[col] = 'location'
    elif 'age' in cl:
        users_col_map[col] = 'age'
users.rename(columns=users_col_map, inplace=True)

print(f"\nBooks columns after rename  : {books.columns.tolist()}")
print(f"Ratings columns after rename: {ratings.columns.tolist()}")
print(f"Users columns after rename  : {users.columns.tolist()}")


# ── Clean Data ────────────────────────────────────────────────
print("\nCleaning data...")

# Strip quotes from values
for col in books.columns:
    if books[col].dtype == object:
        books[col] = books[col].astype(str).str.strip('"').str.strip()

for col in ratings.columns:
    if ratings[col].dtype == object:
        ratings[col] = ratings[col].astype(str).str.strip('"').str.strip()

# Clean books
books = books.dropna(subset=['title','author'])
books['title']     = books['title'].astype(str)
books['author']    = books['author'].astype(str)
books['year']      = pd.to_numeric(books['year'],
                                    errors='coerce')
if 'image_m' not in books.columns:
    books['image_m'] = ''
books['image_m']   = books['image_m'].astype(str)

if 'publisher' not in books.columns:
    books['publisher'] = 'Unknown'
books['publisher'] = books['publisher'].astype(str)

# Clean ratings
ratings['rating'] = pd.to_numeric(ratings['rating'],
                                   errors='coerce')
ratings = ratings.dropna(subset=['rating'])
ratings['rating'] = ratings['rating'].astype(int)

# Keep only explicit ratings (1-10)
ratings = ratings[ratings['rating'] > 0]
ratings = ratings[ratings['rating'] <= 10]

print(f"Books after cleaning   : {books.shape}")
print(f"Ratings after cleaning : {ratings.shape}")


# ── Filter for Quality ────────────────────────────────────────
print("\nFiltering for quality...")

book_counts   = ratings.groupby('ISBN').size()
user_counts   = ratings.groupby('user_id').size()
popular_books = book_counts[book_counts >= 5].index
active_users  = user_counts[user_counts >= 3].index

ratings = ratings[ratings['ISBN'].isin(popular_books)]
ratings = ratings[ratings['user_id'].isin(active_users)]

print(f"Ratings after filtering : {ratings.shape}")
print(f"Unique books            : {ratings['ISBN'].nunique()}")
print(f"Unique users            : {ratings['user_id'].nunique()}")


# ── Sample for Memory Efficiency ─────────────────────────────
if len(ratings) > 5000:
    ratings = ratings.sample(n=5000, random_state=42)
    print(f"Sampled to : {ratings.shape}")


# ── Merge Books with Ratings ──────────────────────────────────
print("\nMerging datasets...")
df = ratings.merge(books, on='ISBN', how='left')
df = df.dropna(subset=['title'])
df = df.reset_index(drop=True)

print(f"Merged dataframe shape : {df.shape}")
print(f"\nSample:\n{df[['title','author','rating']].head()}")


# ── Feature Engineering ───────────────────────────────────────
print("\nFeature engineering...")

scaler            = MinMaxScaler()
df['rating_norm'] = scaler.fit_transform(df[['rating']])

avg_ratings = df.groupby('ISBN')['rating'].agg(
    ['mean','count']).reset_index()
avg_ratings.columns = ['ISBN','avg_rating','num_ratings']

C = avg_ratings['avg_rating'].mean()
m = avg_ratings['num_ratings'].quantile(0.75)
avg_ratings['weighted_rating'] = (
    (avg_ratings['num_ratings'] * avg_ratings['avg_rating']) +
    (m * C)
) / (avg_ratings['num_ratings'] + m)

df = df.merge(avg_ratings, on='ISBN', how='left')
print("Feature engineering done!")


# ── Build Books Master Dataframe ──────────────────────────────
keep_cols = ['ISBN','title','author','year',
             'publisher','image_m',
             'avg_rating','num_ratings','weighted_rating']
keep_cols = [c for c in keep_cols if c in df.columns]

books_df = df.drop_duplicates(subset=['ISBN'])[
    keep_cols].reset_index(drop=True)

if 'publisher' not in books_df.columns:
    books_df['publisher'] = 'Unknown'

print(f"\nBooks master dataframe : {books_df.shape}")
print(f"Columns                : {books_df.columns.tolist()}")


# ── TF-IDF on Book Titles and Authors ────────────────────────
print("\nBuilding TF-IDF matrix...")
books_df['content'] = books_df['title'].astype(str) + \
                      ' ' + books_df['author'].astype(str)

tfidf = TfidfVectorizer(
    max_features = 3000,
    ngram_range  = (1, 2),
    min_df       = 1,
    max_df       = 0.95,
    stop_words   = 'english'
)
tfidf_matrix = tfidf.fit_transform(books_df['content'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")


# ── SVD Matrix Factorization ──────────────────────────────────
print("\nBuilding SVD matrix...")
n_components = min(50,
                   tfidf_matrix.shape[1]-1,
                   tfidf_matrix.shape[0]-1)
svd          = TruncatedSVD(
    n_components=n_components, random_state=42)
svd_matrix   = svd.fit_transform(tfidf_matrix)
print(f"SVD Matrix Shape    : {svd_matrix.shape}")
print(f"Variance Captured   : "
      f"{svd.explained_variance_ratio_.sum()*100:.1f}%")


# ── Save All Models ───────────────────────────────────────────
print("\nSaving models...")
save_path = os.path.dirname(os.path.abspath(__file__))

books_df.to_pickle(os.path.join(save_path, 'books_df.pkl'))
df.to_pickle(os.path.join(save_path, 'df.pkl'))
ratings.to_pickle(os.path.join(save_path, 'ratings.pkl'))

with open(os.path.join(save_path, 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

sp.save_npz(os.path.join(save_path, 'tfidf_matrix.npz'),
            tfidf_matrix)

np.save(os.path.join(save_path, 'svd_matrix.npy'),
        svd_matrix)

print("\n✅ All models saved successfully!")
print(f"Saved to: {save_path}")
print(f"\nFinal Summary:")
print(f"  Total Books   : {len(books_df)}")
print(f"  Total Ratings : {len(ratings)}")
print(f"  Total Users   : {ratings['user_id'].nunique()}")
print(f"  Avg Rating    : {df['rating'].mean():.2f}")
print(f"  Google Books  : "
      f"{'✅ Enabled' if GOOGLE_BOOKS_KEY else '⚠️ Not configured'}")