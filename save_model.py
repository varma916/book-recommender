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
        url      = f"https://www.googleapis.com/books/v1/volumes?q=harry+potter&key={GOOGLE_BOOKS_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✅ Google Books API working!")
        else:
            print(f"⚠️ Google Books API returned: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Google Books API error: {e}")
else:
    print("⚠️ No Google Books API key found")


# ── Download Dataset ──────────────────────────────────────────
print("\nDownloading Book Crossing dataset...")
try:
    url      = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
    response = requests.get(url, timeout=60)
    zip_data = zipfile.ZipFile(io.BytesIO(response.content))
    zip_data.extractall('/tmp/bookcrossing/')
    print("✅ Dataset downloaded!")
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
    print(f"Books   : {books.shape}")
    print(f"Ratings : {ratings.shape}")
    print(f"Users   : {users.shape}")
except Exception as e:
    print(f"Error loading files: {e}")
    raise


# ── Clean Column Names ────────────────────────────────────────
books.columns   = ['ISBN','title','author','year',
                   'publisher','image_s','image_m','image_l']
ratings.columns = ['user_id','ISBN','rating']
users.columns   = ['user_id','location','age']

print("\nColumn names cleaned!")
print("Books columns   :", books.columns.tolist())
print("Ratings columns :", ratings.columns.tolist())
print("Users columns   :", users.columns.tolist())


# ── Clean Data ────────────────────────────────────────────────
print("\nCleaning data...")

# Clean books
books = books.dropna(subset=['title','author'])
books['title']     = books['title'].astype(str)
books['author']    = books['author'].astype(str)
books['year']      = pd.to_numeric(books['year'],
                                    errors='coerce')
books['image_m']   = books['image_m'].astype(str)

# Clean ratings — keep only explicit ratings (1-10)
ratings = ratings[ratings['rating'] > 0]
ratings = ratings[ratings['rating'] <= 10]

print(f"Books after cleaning   : {books.shape}")
print(f"Ratings after cleaning : {ratings.shape}")


# ── Filter for Quality ────────────────────────────────────────
# Keep books with at least 10 ratings
# Keep users with at least 5 ratings
print("\nFiltering for quality...")

book_counts = ratings.groupby('ISBN').size()
user_counts = ratings.groupby('user_id').size()

popular_books = book_counts[book_counts >= 10].index
active_users  = user_counts[user_counts >= 5].index

ratings = ratings[ratings['ISBN'].isin(popular_books)]
ratings = ratings[ratings['user_id'].isin(active_users)]

print(f"Ratings after filtering : {ratings.shape}")
print(f"Unique books            : {ratings['ISBN'].nunique()}")
print(f"Unique users            : {ratings['user_id'].nunique()}")


# ── Sample for Memory Efficiency ─────────────────────────────
# Sample 5000 ratings for Render free tier
if len(ratings) > 5000:
    ratings = ratings.sample(n=5000, random_state=42)
    print(f"Sampled to : {ratings.shape}")


# ── Merge Books with Ratings ──────────────────────────────────
print("\nMerging datasets...")
df = ratings.merge(books, on='ISBN', how='left')
df = df.dropna(subset=['title'])
df = df.reset_index(drop=True)

print(f"Merged dataframe shape : {df.shape}")
print(f"Columns                : {df.columns.tolist()}")
print(f"\nSample:\n{df[['title','author','rating']].head()}")


# ── Feature Engineering ───────────────────────────────────────
print("\nFeature engineering...")

# Normalize ratings
scaler = MinMaxScaler()
df['rating_norm'] = scaler.fit_transform(df[['rating']])

# Average rating per book
avg_ratings = df.groupby('ISBN')['rating'].agg(
    ['mean','count']).reset_index()
avg_ratings.columns = ['ISBN','avg_rating','num_ratings']

# Weighted rating (like IMDB formula)
C = avg_ratings['avg_rating'].mean()
m = avg_ratings['num_ratings'].quantile(0.75)
avg_ratings['weighted_rating'] = (
    (avg_ratings['num_ratings'] * avg_ratings['avg_rating']) +
    (m * C)
) / (avg_ratings['num_ratings'] + m)

df = df.merge(avg_ratings, on='ISBN', how='left')
print("Feature engineering done!")


# ── Build Books Master Dataframe ──────────────────────────────
books_df = df.drop_duplicates(subset=['ISBN'])[
    ['ISBN','title','author','year','publisher',
     'image_m','avg_rating','num_ratings',
     'weighted_rating']
].reset_index(drop=True)

print(f"\nBooks master dataframe : {books_df.shape}")


# ── TF-IDF on Book Titles and Authors ────────────────────────
print("\nBuilding TF-IDF matrix...")
books_df['content'] = books_df['title'] + ' ' + \
                      books_df['author']

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
n_components = min(50, tfidf_matrix.shape[1]-1,
                   tfidf_matrix.shape[0]-1)
svd          = TruncatedSVD(n_components=n_components,
                             random_state=42)
svd_matrix   = svd.fit_transform(tfidf_matrix)
print(f"SVD Matrix Shape: {svd_matrix.shape}")
print(f"Variance Captured: {svd.explained_variance_ratio_.sum()*100:.1f}%")


# ── Save All Models ───────────────────────────────────────────
print("\nSaving models...")
save_path = os.path.dirname(os.path.abspath(__file__))

# Save dataframes
books_df.to_pickle(os.path.join(save_path, 'books_df.pkl'))
df.to_pickle(os.path.join(save_path, 'df.pkl'))
ratings.to_pickle(os.path.join(save_path, 'ratings.pkl'))

# Save TF-IDF
with open(os.path.join(save_path, 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

# Save sparse matrix
sp.save_npz(os.path.join(save_path, 'tfidf_matrix.npz'),
            tfidf_matrix)

# Save SVD matrix
np.save(os.path.join(save_path, 'svd_matrix.npy'),
        svd_matrix)

print("\n✅ All models saved successfully!")
print(f"Saved to: {save_path}")
print(f"\nFinal Summary:")
print(f"  Total Books   : {len(books_df)}")
print(f"  Total Ratings : {len(ratings)}")
print(f"  Total Users   : {ratings['user_id'].nunique()}")
print(f"  Avg Rating    : {df['rating'].mean():.2f}")
print(f"  Google Books  : {'✅ Enabled' if GOOGLE_BOOKS_KEY else '⚠️ Not configured'}")