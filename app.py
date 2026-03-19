import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title = "Book Recommender",
    page_icon  = "📚",
    layout     = "wide"
)

FASTAPI_URL = "https://book-recommender-api-idm1.onrender.com"

st.title("Book Recommendation System")
st.markdown("*Powered by TF-IDF + SVD + Collaborative Filtering + Google Books API*")
st.markdown("---")

try:
    stats = requests.get(
        f"{FASTAPI_URL}/stats", timeout=30).json()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Books",   stats['total_books'])
    col2.metric("Total Ratings", stats['total_ratings'])
    col3.metric("Total Users",   stats['total_users'])
    col4.metric("Avg Rating",    stats['avg_rating'])
    col5.metric("Google Books",  stats['google_books'])
    st.markdown("---")
except:
    st.warning("Could not connect to FastAPI!")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Search by Title",
    "Search by Author / Publisher / Year / Genre",
    "User Recommendations",
    "Popular Books",
    "Live Book Search",
    "Stats"
])


def show_books(books):
    if not books:
        st.error("No books found!")
        return
    cols = st.columns(3)
    for i, book in enumerate(books):
        with cols[i % 3]:
            img = book.get('image_m') or \
                  book.get('cover') or ''
            if img and img != 'nan' and \
               img.startswith('http'):
                try:
                    st.image(img, use_column_width=True)
                except:
                    st.image(
                        "https://via.placeholder.com/200x300?text=No+Cover",
                        use_column_width=True)
            else:
                st.image(
                    "https://via.placeholder.com/200x300?text=No+Cover",
                    use_column_width=True)

            title  = str(book.get('title', ''))[:50]
            author = book.get('author') or \
                     book.get('authors', '')
            st.markdown(f"**{title}**")
            st.caption(f"by {author}")

            rating = book.get('avg_rating') or \
                     book.get('rating', 0)
            if rating:
                st.caption(f"Rating: {rating}")

            rating_count = book.get('rating_count', 0)
            if rating_count:
                st.caption(f"Reviews: {rating_count}")

            year = book.get('year', '')
            if year and str(year) != 'nan':
                st.caption(f"Year: {year}")

            publisher = book.get('publisher', '')
            if publisher and publisher != 'Unknown':
                st.caption(f"Publisher: {publisher}")

            pages = book.get('pages', 0)
            if pages:
                st.caption(f"Pages: {pages}")

            published = book.get('published', '')
            if published:
                st.caption(f"Published: {published}")

            category = book.get('category', '')
            if category and category != 'Unknown':
                st.caption(f"Category: {category}")

            description = book.get('description', '')
            if description:
                st.caption(
                    str(description)[:120] + '...')

            preview = book.get('preview_url', '')
            if preview:
                st.markdown(
                    f"[Preview on Google Books]({preview})")

            score = book.get('score_val', 0)
            if score:
                st.caption(
                    f"Score: {float(score):.3f}")

            num_ratings = book.get('num_ratings', 0)
            if num_ratings:
                st.caption(f"Ratings: {int(num_ratings)}")

            st.markdown("---")


with tab1:
    st.subheader("Find Similar Books by Title")
    st.markdown(
        "Enter any book title to find similar books "
        "using our ML recommendation engine")
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        book_title = st.text_input(
            "Enter Book Title:",
            placeholder="e.g. Harry Potter, Da Vinci Code")
    with col2:
        method = st.selectbox(
            "Recommendation Method",
            ["hybrid", "tfidf", "svd", "popularity"],
            format_func=lambda x: {
                "hybrid"    : "Hybrid (Best)",
                "tfidf"     : "TF-IDF Content Based",
                "svd"       : "SVD Semantic",
                "popularity": "Popularity Based"
            }[x])
    with col3:
        top_n = st.slider("Results", 3, 10, 6)

    if st.button("Get Recommendations", key="title_btn"):
        if not book_title:
            st.warning("Please enter a book title!")
        else:
            with st.spinner("Finding similar books..."):
                try:
                    response = requests.post(
                        f"{FASTAPI_URL}/recommend",
                        json={
                            "book_title": book_title,
                            "method"    : method,
                            "top_n"     : top_n
                        },
                        timeout=30)
                    data = response.json()
                    if data['total_results'] == 0:
                        st.error(
                            "No recommendations found! "
                            "Try another title.")
                    else:
                        st.success(
                            f"Found {data['total_results']} "
                            f"books using "
                            f"{data['method'].upper()}!")
                        st.markdown("---")
                        show_books(data['recommendations'])
                except Exception as e:
                    st.error(f"Error: {e}")


with tab2:
    st.subheader(
        "Search by Author, Publisher, Year or Genre")
    search_type = st.radio(
        "Search By:",
        ["Author", "Publisher", "Year", "Genre"],
        horizontal=True)

    col1, col2 = st.columns([3, 1])

    if search_type == "Author":
        with col1:
            author_name = st.text_input(
                "Enter Author Name:",
                placeholder="e.g. Stephen King, J.K. Rowling")
        with col2:
            author_top_n = st.slider(
                "Results", 3, 10, 6, key="author_slider")
        if st.button("Search by Author", key="author_btn"):
            if not author_name:
                st.warning("Please enter an author name!")
            else:
                with st.spinner(
                        f"Searching books by "
                        f"{author_name}..."):
                    try:
                        response = requests.get(
                            f"{FASTAPI_URL}/author-search",
                            params={
                                "author_name": author_name,
                                "top_n": author_top_n
                            },
                            timeout=30)
                        data  = response.json()
                        books = data['books']
                        if not books:
                            st.error(
                                f"No books found by "
                                f"{author_name}! "
                                f"Try a different name.")
                        else:
                            st.success(
                                f"Found {len(books)} books "
                                f"by {author_name}!")
                            st.markdown("---")
                            show_books(books)
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif search_type == "Publisher":
        with col1:
            publisher_name = st.text_input(
                "Enter Publisher Name:",
                placeholder="e.g. Penguin, HarperCollins, Oxford")
        with col2:
            pub_top_n = st.slider(
                "Results", 3, 10, 6, key="pub_slider")
        if st.button("Search by Publisher",
                     key="publisher_btn"):
            if not publisher_name:
                st.warning("Please enter a publisher name!")
            else:
                with st.spinner(
                        f"Searching books from "
                        f"{publisher_name}..."):
                    try:
                        response = requests.get(
                            f"{FASTAPI_URL}/publisher-search",
                            params={
                                "publisher_name": publisher_name,
                                "top_n": pub_top_n
                            },
                            timeout=30)
                        data  = response.json()
                        books = data['books']
                        if not books:
                            st.error(
                                f"No books found from "
                                f"{publisher_name}!")
                        else:
                            st.success(
                                f"Found {len(books)} books "
                                f"from {publisher_name}!")
                            st.markdown("---")
                            show_books(books)
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif search_type == "Year":
        with col1:
            year = st.number_input(
                "Enter Publication Year:",
                min_value=1800,
                max_value=2024,
                value=2000,
                step=1)
        with col2:
            year_top_n = st.slider(
                "Results", 3, 10, 6, key="year_slider")
        if st.button("Search by Year", key="year_btn"):
            with st.spinner(
                    f"Searching books from {year}..."):
                try:
                    response = requests.get(
                        f"{FASTAPI_URL}/year-search",
                        params={
                            "year" : year,
                            "top_n": year_top_n
                        },
                        timeout=30)
                    data  = response.json()
                    books = data['books']
                    if not books:
                        st.error(
                            f"No books found from {year}! "
                            f"Try 1995, 1998, 2000, 2001, "
                            f"2002 or 2003.")
                    else:
                        st.success(
                            f"Found {len(books)} books "
                            f"from {year}!")
                        st.markdown("---")
                        show_books(books)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif search_type == "Genre":
        with col1:
            genre = st.selectbox(
                "Select Genre:",
                ["Fiction", "Non-Fiction", "Mystery",
                 "Romance", "Science Fiction", "Fantasy",
                 "Biography", "History", "Self Help",
                 "Horror", "Thriller", "Children",
                 "Poetry", "Travel", "Cooking",
                 "Business", "Technology", "Sports"])
        with col2:
            genre_top_n = st.slider(
                "Results", 3, 10, 6, key="genre_slider")
        if st.button("Search by Genre", key="genre_btn"):
            with st.spinner(
                    f"Searching {genre} books from "
                    f"Google Books..."):
                try:
                    response = requests.get(
                        f"{FASTAPI_URL}/genre-search",
                        params={
                            "genre" : genre,
                            "top_n" : genre_top_n
                        },
                        timeout=30)
                    data  = response.json()
                    books = data['books']
                    if not books:
                        st.error(
                            f"No {genre} books found!")
                    else:
                        st.success(
                            f"Found {len(books)} {genre} "
                            f"books from Google Books!")
                        st.markdown("---")
                        show_books(books)
                except Exception as e:
                    st.error(f"Error: {e}")


with tab3:
    st.subheader("Personalized Book Recommendations")
    st.markdown(
        "Get book recommendations based on your "
        "reading history using Collaborative Filtering")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.number_input(
            "Enter User ID:",
            min_value=1,
            value=276725,
            step=1)
    with col2:
        user_top_n = st.slider(
            "Results", 3, 10, 6, key="user_slider")
    if st.button("Get My Recommendations",
                 key="user_btn"):
        with st.spinner(
                "Loading personalized recommendations..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/user-recommend",
                    json={
                        "user_id": user_id,
                        "top_n"  : user_top_n
                    },
                    timeout=30)
                data = response.json()
                if data['total_results'] == 0:
                    st.error("No recommendations found!")
                else:
                    st.success(
                        f"Found {data['total_results']} "
                        f"recommendations for "
                        f"User {user_id}!")
                    st.markdown("---")
                    show_books(data['recommendations'])
            except Exception as e:
                st.error(f"Error: {e}")


with tab4:
    st.subheader("Most Popular Books")
    st.markdown(
        "Top books ranked by weighted rating formula "
        "(similar to IMDB rating system)")
    pop_top_n = st.slider(
        "Number of Books", 3, 10, 6, key="pop_slider")
    if st.button("Load Popular Books", key="pop_btn"):
        with st.spinner("Loading popular books..."):
            try:
                response = requests.get(
                    f"{FASTAPI_URL}/popular",
                    params={"top_n": pop_top_n},
                    timeout=30)
                data  = response.json()
                books = data['books']
                if not books:
                    st.error("No books found!")
                else:
                    st.success(
                        f"Top {len(books)} most "
                        f"popular books!")
                    st.markdown("---")
                    show_books(books)
            except Exception as e:
                st.error(f"Error: {e}")


with tab5:
    st.subheader("Live Book Search")
    st.markdown(
        "Search any book in the world using "
        "Google Books API - Real time results "
        "sorted by relevance!")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search for any book:",
            placeholder=(
                "e.g. Machine Learning, "
                "Python programming, Cricket"))
    with col2:
        max_results = st.slider(
            "Max Results", 5, 20, 10)
    if st.button("Search Google Books",
                 key="live_btn"):
        if not search_query:
            st.warning("Please enter a search query!")
        else:
            with st.spinner(
                    f"Searching Google Books for "
                    f"'{search_query}'..."):
                try:
                    response = requests.get(
                        f"{FASTAPI_URL}/live-search",
                        params={
                            "query"      : search_query,
                            "max_results": max_results
                        },
                        timeout=30)
                    data  = response.json()
                    books = data['books']
                    if not books:
                        st.error("No books found!")
                    else:
                        st.success(
                            f"Found {len(books)} books "
                            f"from Google Books sorted "
                            f"by relevance!")
                        st.markdown("---")
                        show_books(books)
                except Exception as e:
                    st.error(f"Error: {e}")


with tab6:
    st.subheader("Dataset Statistics")
    try:
        stats = requests.get(
            f"{FASTAPI_URL}/stats", timeout=30).json()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Books",
                      stats['total_books'])
            st.metric("Total Ratings",
                      stats['total_ratings'])
        with col2:
            st.metric("Total Users",
                      stats['total_users'])
            st.metric("Avg Rating",
                      stats['avg_rating'])
        with col3:
            st.metric("Top Author",
                      stats['top_author'])
            st.metric("Google Books",
                      stats['google_books'])
        st.markdown("---")
        st.info(
            "This system uses the Book Crossing Dataset "
            "with over 1 million real book ratings. "
            "Uses TF-IDF content filtering, SVD matrix "
            "factorization and collaborative filtering "
            "to recommend books."
        )
        st.markdown("---")
        st.markdown("**Available Search Options**")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Search by Book Title")
            st.success("Search by Author Name")
            st.success("Search by Publisher")
        with col2:
            st.success("Search by Publication Year")
            st.success("Search by Genre (Live)")
            st.success("Live Google Books Search")
    except Exception as e:
        st.error(f"Error: {e}")


st.markdown("---")
st.markdown(
    "*Built with FastAPI + Streamlit | "
    "Dataset: Book Crossing | "
    "Live Search: Google Books API | "
    "Algorithms: TF-IDF + SVD + Collaborative Filtering*"
)