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
        url      = (
            f"https://www.googleapis.com/books/v1/volumes?"
            f"q={query}&maxResults={max_results}"
            f"&key={GOOGLE_BOOKS_KEY}"
        )
        response = req.get(url, timeout=10)
        if response.status_code != 200:
            return []
        data  = response.json()
        books = []
        for item in data.get('items', []):
            info = item.get('volumeInfo', {})
            books.append({
                'title'      : info.get('title', ''),
                'authors'    : ', '.join(
                    info.get('authors', ['Unknown'])),
                'description': info.get(
                    'description', '')[:200] + '...'
                    if info.get('description') else '',
                'cover'      : info.get(
                    'imageLinks', {}).get('thumbnail', ''),
                'rating'     : info.get('averageRating', 0),
                'pages'      : info.get('pageCount', 0),
                'published'  : info.get('publishedDate', ''),
                'category'   : ', '.join(
                    info.get('categories', ['Unknown'])),
                'publisher'  : info.get('publisher', 'Unknown'),
                'preview_url': info.get('previewLink', '')
            })
        return books
    except Exception as e:
        print(f"Google Books error: {e}")
        return []


# ── Content Based Recommender (TF-IDF) ───────────────────────
def content_based_recommend(book_title, top_n=10):
    matches = books_df[books_df['title'].str.contains(
        book_title, case=False, na=False)]
    if matches.empty:
        return []
    idx        = matches.index[0]
    query_vec  = tfidf_matrix[idx]
    sim_scores = cosine_similarity(
        query_vec, tfidf_matrix).flatten()
    sim_scores[idx] = 0
    top_indices = sim_scores.argsort()[::-1][:top_n]
    result      = books_df.iloc[top_indices].copy()
    result['score_val'] = sim_scores[top_indices]
    result['method']    = 'TF-IDF Content'
    return result[[
        'ISBN','title','author','year',
        'image_m','avg_rating','num_ratings',
        'score_val','method']].to_dict('records')


# ── SVD Based Recommender ─────────────────────────────────────
def svd_based_recommend(book_title, top_n=10):
    matches = books_df[books_df['title'].str.contains(
        book_title, case=False, na=False)]
    if matches.empty:
        return []
    idx        = matches.index[0]
    query_vec  = svd_matrix[idx].reshape(1, -1)
    sim_scores = cosine_similarity(
        query_vec, svd_matrix).flatten()
    sim_scores[idx] = 0
    top_indices = sim_scores.argsort()[::-1][:top_n]
    result      = books_df.iloc[top_indices].copy()
    result['score_val'] = sim_scores[top_indices]
    result['method']    = 'SVD'
    return result[[
        'ISBN','title','author','year',
        'image_m','avg_rating','num_ratings',
        'score_val','method']].to_dict('records')


# ── Popularity Based Recommender ──────────────────────────────
def popularity_based_recommend(top_n=10):
    result = books_df.nlargest(top_n, 'weighted_rating')[
        ['ISBN','title','author','year',
         'image_m','avg_rating','num_ratings',
         'weighted_rating']].copy()
    result['score_val'] = result['weighted_rating']
    result['method']    = 'Popularity'
    return result.to_dict('records')


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
    result = book_scores.merge(books_df, on='ISBN', how='left')
    result = result.dropna(subset=['title'])
    result['method'] = 'Collaborative'
    return result[[
        'ISBN','title','author','year',
        'image_m','avg_rating','num_ratings',
        'score_val','method']].to_dict('records')


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
    result['method']    = 'Hybrid'
    result = result.sort_values('score_val', ascending=False)
    return result[[
        'ISBN','title','author','year',
        'image_m','avg_rating','num_ratings',
        'score_val','method']].to_dict('records')


# ── Search by Author ──────────────────────────────────────────
def search_by_author(author_name, top_n=10):
    matches = books_df[books_df['author'].str.contains(
        author_name, case=False, na=False)]
    if matches.empty:
        return []
    result = matches.nlargest(top_n, 'avg_rating')[
        ['ISBN','title','author','year',
         'image_m','avg_rating','num_ratings']].copy()
    result['score_val'] = result['avg_rating'] / 10
    result['method']    = 'Author Search'
    return result.to_dict('records')


# ── Search by Publisher ───────────────────────────────────────
def search_by_publisher(publisher_name, top_n=10):
    matches = books_df[books_df['publisher'].str.contains(
        publisher_name, case=False, na=False)] \
        if 'publisher' in books_df.columns else pd.DataFrame()
    if matches.empty:
        return []
    result = matches.nlargest(top_n, 'avg_rating')[
        ['ISBN','title','author','year',
         'image_m','avg_rating','num_ratings']].copy()
    result['score_val'] = result['avg_rating'] / 10
    result['method']    = 'Publisher Search'
    return result.to_dict('records')


# ── Search by Year ────────────────────────────────────────────
def search_by_year(year, top_n=10):
    matches = books_df[books_df['year'] == year]
    if matches.empty:
        return []
    result = matches.nlargest(top_n, 'avg_rating')[
        ['ISBN','title','author','year',
         'image_m','avg_rating','num_ratings']].copy()
    result['score_val'] = result['avg_rating'] / 10
    result['method']    = 'Year Search'
    return result.to_dict('records')


# ── Search by Genre (via Google Books) ───────────────────────
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
    return {"total_results": len(results), "books": results}


@app.get("/live-search")
def live_search(query: str, max_results: int = 10):
    results = get_google_books(query, max_results)
    return {
        "query"        : query,
        "total_results": len(results),
        "source"       : "Google Books API",
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
    return {
        "total_books"  : len(books_df),
        "total_ratings": len(ratings),
        "total_users"  : ratings['user_id'].nunique(),
        "avg_rating"   : round(df['rating'].mean(), 2),
        "top_author"   : books_df.groupby('author')[
            'num_ratings'].sum().idxmax(),
        "google_books" : "Enabled"
                         if GOOGLE_BOOKS_KEY
                         else "Not configured"
    }