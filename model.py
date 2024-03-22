from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Hàm recommend_products
def recommend_products(book_title, data, top_n=5):
    # Biểu diễn tiêu đề sách dưới dạng vector bằng TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['title'])
    
    # Tính toán ma trận tương đồng cosine giữa các tiêu đề sách
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Tìm index của cuốn sách có tên gần giống với tên được nhập vào
    book_indices = [i for i, title in enumerate(data['title']) if book_title.lower() in title.lower()]
    if not book_indices:
        print("Không tìm thấy cuốn sách phù hợp.")
        return None
    
    # Lấy thông tin của cuốn sách đã nhập
    book_info = data.iloc[book_indices[0]]
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[['current_price', 'pages', 'avg_rating']] = scaler.fit_transform(data[['current_price', 'pages', 'avg_rating']])
    book_info_scaled = data_scaled.iloc[book_indices[0]]
    
    # Lọc lại các cuốn sách để chỉ giữ lại những cuốn sách có cùng thể loại và có điểm tương đồng gần giống
    recommended_indices = []
    for i, sim_scores in enumerate(similarity_matrix[book_indices[0]]):
        if data.iloc[i]['category'] == book_info['category'] and i != book_indices[0]:
            similarity_score = sim_scores
            price_similarity = 1 - abs(book_info_scaled['current_price'] - data_scaled.iloc[i]['current_price'])
            page_similarity = 1 - abs(book_info_scaled['pages'] - data_scaled.iloc[i]['pages'])
            rating_similarity = 1 - abs(book_info_scaled['avg_rating'] - data_scaled.iloc[i]['avg_rating'])
            
            # Tính tổng điểm tương đồng với trọng số, có thể điều chỉnh để phản ánh đúng mức độ quan trọng của từng tiêu chí
            total_similarity = similarity_score * 0.5 + price_similarity * 0.2 + page_similarity * 0.2 + rating_similarity * 0.1
            
            recommended_indices.append((i, total_similarity))
    
    # Sắp xếp các cuốn sách theo điểm tương đồng giảm dần
    recommended_indices.sort(key=lambda x: x[1], reverse=True)  # Chú ý đổi thành reverse=True để lấy điểm cao nhất
    
    # Lấy top_n cuốn sách được đề xuất
    recommended_books = [data.iloc[idx] for idx, _ in recommended_indices[:top_n]]
    
    return recommended_books

