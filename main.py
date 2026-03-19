# main.py - FastAPI Backend for Book Recommendation System

import os
import pickle
import subprocess
import pandas as pd
import numpy as np
import requests as req
import scipy.sparse as sp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

# ── Load Models ───────────────────────────────────────────────
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

def load_models():
    global books_df, df, ratings, tfidf_matrix, svd_matrix
    print("Loading models...")
    try:
        books_df     = pd.read_pickle(
            os.path.join(MODEL_PATH, 'books_df.pkl'))
        df           = pd.read_pickle(
            os.path.join(MODEL_PATH, 'df.pkl'))
        ratings      = pd.read_pickle(
            os.path.join(MODEL_PATH, 'ratings.pkl'))
        tfidf_matrix = sp.load_npz(
            os.path.join(MODEL_PATH, 'tfidf_matrix.npz'))
        svd_matrix   = np.load(
            os.path.join(MODEL_PATH, 'svd_matrix.npy'))
        print("✅ Models loaded successfully!")
        print(f"  Books   : {len(books_df)}")
        print(f"  Ratings : {len(ratings)}")
        print(f"  Users   : {ratings['user_id'].nunique()}")
    except Exception as e:
        print(f"Models not found: {e}")
        print("Running save_model.py...")
        subprocess.run(
            ['python',
             os.path.join(MODEL_PATH, 'save_model.py')],
            check=True)
        books_df     = pd.read_pickle(
            os.path.join(MODEL_PATH, 'books_df.pkl'))
        df           = pd.read_pickle(
            os.path.join(MODEL_PATH, 'df.pkl'))
        ratings      = pd.read_pickle(
            os.path.join(MODEL_PATH, 'ratings.pkl'))
        tfidf_matrix = sp.load_npz(
            os.path.join(MODEL_PATH, 'tfidf_matrix.npz'))
        svd_matrix   = np.load(
            os.path.join(MODEL_PATH, 'svd_matrix.npy'))
        print("✅ Models loaded after regeneration!")

load_models()

# ── Google Books API Key ──────────────────────────────────────
GOOGLE_BOOKS_KEY = os.getenv('GOOGLE_BOOKS_KEY', '')

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title       = "Book Recommendation API",
    description = "Recommends books using TF-IDF + SVD + Collaborative Filtering + Google Books API",
    version     = "2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"]
)


# ── Request Models ────────────────────────────────────────────
class BookRecommendRequest(BaseModel):
    book_title : str
    method     : str = "hybrid"
    top_n      : int = 10

class UserRecommendRequest(BaseModel):
    user_id : int
    top_n   : int = 10


# ── Google Books API ──────────────────────────────────────────
def get_google_books(query, max_results=10):
    if not GOOGLE_BOOKS_KEY:
        return []
    try:
        # Fetch 3x more so we can filter and return best ones
        fetch_count = max_results * 3

        url      = (
            f"https://www.googleapis.com/books/v1/volumes?"
            f"q={query}"
            f"&maxResults={min(fetch_count, 40)}"
            f"&orderBy=relevance"
            f"&printType=books"
            f"&langRestrict=en"
            f"&key={GOOGLE_BOOKS_KEY}"
        )
        response = req.get(url, timeout=10)
        if response.status_code != 200:
            return []
        data  = response.json()
        books = []
        for item in data.get('items', []):
            info = item.get('volumeInfo', {})

            # Skip books without title or authors
            if not info.get('title'):
                continue
            if not info.get('authors'):
                continue

            # Get cover image
            image_links = info.get('imageLinks', {})
            cover = (
                image_links.get('thumbnail') or
                image_links.get('smallThumbnail') or
                ''
            )

            # Replace http with https for cover URLs
            if cover.startswith('http://'):
                cover = cover.replace('http://', 'https://')

            # Calculate relevance score
            has_cover       = 1 if cover else 0
            has_description = 1 if info.get('description') else 0
            has_rating      = 1 if info.get('averageRating') else 0
            rating          = info.get('averageRating', 0)
            rating_count    = info.get('ratingsCount', 0)

            relevance_score = (
                has_cover       * 2 +
                has_description * 1 +
                has_rating      * 2 +
                (rating / 5)    * 3 +
                min(rating_count / 1000, 2)
            )

            books.append({
                'title'          : info.get('title', ''),
                'authors'        : ', '.join(
                    info.get('authors', ['Unknown'])),
                'description'    : info.get(
                    'description', '')[:250] + '...'
                    if info.get('description') else '',
                'cover'          : cover,
                'rating'         : info.get('averageRating', 0),
                'rating_count'   : info.get('ratingsCount', 0),
                'pages'          : info.get('pageCount', 0),
                'published'      : info.get('publishedDate', ''),
                'category'       : ', '.join(
                    info.get('categories', ['Unknown'])),
                'publisher'      : info.get('publisher', 'Unknown'),
                'preview_url'    : info.get('previewLink', ''),
                'relevance_score': relevance_score
            })

        # Sort by relevance score
        books = sorted(books,
                       key=lambda x: x['relevance_score'],
                       reverse=True)

        return books[:max_results]

    except Exception as e:
        print(f"Google Books error: {e}")
        return []


# ── Helper — Format Book Result ───────────────────────────────
def format_book(row, score_val=0, method=''):
    return {
        'ISBN'       : str(row.get('ISBN', '')),
        'title'      : str(row.get('title', '')),
        'author'     : str(row.get('author', '')),
        'year'       : str(row.get('year', '')),
        'publisher'  : str(row.get('publisher', 'Unknown')),
        'image_m'    : str(row.get('image_m', '')),
        'avg_rating' : float(row.get('avg_rating', 0)),
        'num_ratings': int(row.get('num_ratings', 0)),
        'score_val'  : round(float(score_val), 4),
        'method'     : method
    }


# ── Content Based Recommender (TF-IDF) ───────────────────────
def content_based_recommend(book_title, top_n=10):
    matches = books_df[books_df['title'].str.contains(
        book_title, case=False, na=False)]
    if matches.empty:
        return []
    idx         = matches.index[0]
    query_vec   = tfidf_matrix[idx]
    sim_scores  = cosine_similarity(
        query_vec, tfidf_matrix).flatten()
    sim_scores[idx] = 0
    top_indices = sim_scores.argsort()[::-1][:top_n]
    results     = []
    for idx2 in top_indices:
        row = books_df.iloc[idx2]
        results.append(format_book(
            row, sim_scores[idx2], 'TF-IDF Content'))
    return results


# ── SVD Based Recommender ─────────────────────────────────────
def svd_based_recommend(book_title, top_n=10):
    matches = books_df[books_df['title'].str.contains(
        book_title, case=False, na=False)]
    if matches.empty:
        return []
    idx         = matches.index[0]
    query_vec   = svd_matrix[idx].reshape(1, -1)
    sim_scores  = cosine_similarity(
        query_vec, svd_matrix).flatten()
    sim_scores[idx] = 0
    top_indices = sim_scores.argsort()[::-1][:top_n]
    results     = []
    for idx2 in top_indices:
        row = books_df.iloc[idx2]
        results.append(format_book(
            row, sim_scores[idx2], 'SVD'))
    return results


# ── Popularity Based Recommender ──────────────────────────────
def popularity_based_recommend(top_n=10):
    top = books_df.nlargest(top_n, 'weighted_rating')
    return [format_book(row, row['weighted_rating'],
                        'Popularity')
            for _, row in top.iterrows()]


# ── Collaborative Filtering ───────────────────────────────────
def collaborative_recommend(user_id, top_n=10):
    user_ratings = ratings[ratings['user_id'] == user_id]
    if user_ratings.empty:
        return popularity_based_recommend(top_n)
    rated_isbns   = user_ratings['ISBN'].tolist()
    similar_users = ratings[
        ratings['ISBN'].isin(rated_isbns) &
        (ratings['user_id'] != user_id)
    ]['user_id'].value_counts().head(20).index.tolist()
    if not similar_users:
        return popularity_based_recommend(top_n)
    sim_user_ratings = ratings[
        ratings['user_id'].isin(similar_users) &
        ~ratings['ISBN'].isin(rated_isbns) &
        (ratings['rating'] >= 7)
    ]
    if sim_user_ratings.empty:
        return popularity_based_recommend(top_n)
    book_scores = sim_user_ratings.groupby(
        'ISBN')['rating'].agg(['mean','count']).reset_index()
    book_scores.columns = ['ISBN','mean_rating','count']
    book_scores['score_val'] = (
        book_scores['mean_rating'] *
        np.log1p(book_scores['count'])
    )
    book_scores = book_scores.sort_values(
        'score_val', ascending=False).head(top_n)
    result  = book_scores.merge(
        books_df, on='ISBN', how='left')
    result  = result.dropna(subset=['title'])
    results = []
    for _, row in result.iterrows():
        results.append(format_book(
            row, row['score_val'], 'Collaborative'))
    return results


# ── Hybrid Recommender ────────────────────────────────────────
def hybrid_recommend(book_title, top_n=10):
    tfidf_recs = content_based_recommend(book_title, top_n=50)
    svd_recs   = svd_based_recommend(book_title, top_n=50)
    pop_recs   = popularity_based_recommend(top_n=50)
    score_dict = {}
    for rec in tfidf_recs:
        isbn = rec['ISBN']
        score_dict[isbn] = score_dict.get(isbn, 0) + \
                           0.4 * rec['score_val']
    for rec in svd_recs:
        isbn = rec['ISBN']
        score_dict[isbn] = score_dict.get(isbn, 0) + \
                           0.3 * rec['score_val']
    for rec in pop_recs:
        isbn = rec['ISBN']
        score_dict[isbn] = score_dict.get(isbn, 0) + \
                           0.3 * rec['score_val']
    top_isbns      = sorted(score_dict.items(),
                            key=lambda x: x[1],
                            reverse=True)[:top_n]
    top_isbns_list = [i[0] for i in top_isbns]
    scores_list    = [i[1] for i in top_isbns]
    result         = books_df[
        books_df['ISBN'].isin(top_isbns_list)].copy()
    score_map           = dict(zip(top_isbns_list, scores_list))
    result['score_val'] = result['ISBN'].map(score_map)
    result = result.sort_values('score_val', ascending=False)
    results = []
    for _, row in result.iterrows():
        results.append(format_book(
            row, row['score_val'], 'Hybrid'))
    return results


# ── Search by Author ──────────────────────────────────────────
def search_by_author(author_name, top_n=10):
    matches = books_df[books_df['author'].str.contains(
        author_name, case=False, na=False)]
    if matches.empty:
        return []
    top = matches.nlargest(top_n, 'avg_rating')
    return [format_book(row, row['avg_rating']/10,
                        'Author Search')
            for _, row in top.iterrows()]


# ── Search by Publisher ───────────────────────────────────────
def search_by_publisher(publisher_name, top_n=10):
    if 'publisher' not in books_df.columns:
        return []
    matches = books_df[books_df['publisher'].str.contains(
        publisher_name, case=False, na=False)]
    if matches.empty:
        return []
    top = matches.nlargest(top_n, 'avg_rating')
    return [format_book(row, row['avg_rating']/10,
                        'Publisher Search')
            for _, row in top.iterrows()]


# ── Search by Year ────────────────────────────────────────────
def search_by_year(year, top_n=10):
    matches = books_df[books_df['year'] == year]
    if matches.empty:
        return []
    top = matches.nlargest(top_n, 'avg_rating')
    return [format_book(row, row['avg_rating']/10,
                        'Year Search')
            for _, row in top.iterrows()]


# ── Search by Genre (Google Books) ───────────────────────────
def search_by_genre(genre, top_n=10):
    return get_google_books(
        f"subject:{genre}", max_results=top_n)


# ── API Routes ────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message"  : "Book Recommendation API is running!",
        "version"  : "2.0",
        "endpoints": [
            "/recommend", "/user-recommend",
            "/popular", "/live-search", "/stats",
            "/author-search", "/publisher-search",
            "/year-search", "/genre-search"
        ]
    }


@app.post("/recommend")
def get_recommendations(request: BookRecommendRequest):
    method = request.method.lower()
    if method == "tfidf":
        results = content_based_recommend(
            request.book_title, request.top_n)
    elif method == "svd":
        results = svd_based_recommend(
            request.book_title, request.top_n)
    elif method == "popularity":
        results = popularity_based_recommend(request.top_n)
    else:
        results = hybrid_recommend(
            request.book_title, request.top_n)
    return {
        "method"         : method,
        "query"          : request.book_title,
        "total_results"  : len(results),
        "recommendations": results
    }


@app.post("/user-recommend")
def get_user_recommendations(request: UserRecommendRequest):
    results = collaborative_recommend(
        request.user_id, request.top_n)
    return {
        "user_id"        : request.user_id,
        "total_results"  : len(results),
        "recommendations": results
    }


@app.get("/popular")
def get_popular(top_n: int = 10):
    results = popularity_based_recommend(top_n)
    return {
        "total_results": len(results),
        "books"        : results
    }


@app.get("/live-search")
def live_search(query: str, max_results: int = 10):
    results = get_google_books(query, max_results)
    return {
        "query"        : query,
        "total_results": len(results),
        "source"       : "Google Books API - Sorted by Relevance",
        "books"        : results
    }


@app.get("/author-search")
def author_search(author_name: str, top_n: int = 10):
    results = search_by_author(author_name, top_n)
    return {
        "author"       : author_name,
        "total_results": len(results),
        "books"        : results
    }


@app.get("/publisher-search")
def publisher_search(publisher_name: str, top_n: int = 10):
    results = search_by_publisher(publisher_name, top_n)
    return {
        "publisher"    : publisher_name,
        "total_results": len(results),
        "books"        : results
    }


@app.get("/year-search")
def year_search(year: int, top_n: int = 10):
    results = search_by_year(year, top_n)
    return {
        "year"         : year,
        "total_results": len(results),
        "books"        : results
    }


@app.get("/genre-search")
def genre_search(genre: str, top_n: int = 10):
    results = search_by_genre(genre, top_n)
    return {
        "genre"        : genre,
        "total_results": len(results),
        "source"       : "Google Books API",
        "books"        : results
    }


@app.get("/stats")
def get_stats():
    top_author = "Unknown"
    try:
        top_author = books_df.groupby('author')[
            'num_ratings'].sum().idxmax()
    except:
        pass
    return {
        "total_books"  : len(books_df),
        "total_ratings": len(ratings),
        "total_users"  : int(ratings['user_id'].nunique()),
        "avg_rating"   : round(float(df['rating'].mean()), 2),
        "top_author"   : top_author,
        "google_books" : "Enabled"
                         if GOOGLE_BOOKS_KEY
                         else "Not configured"
    }