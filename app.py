import streamlit as st
import joblib
import pandas as pd
from model import recommend_products 
st.header('Book Recommender System Using Machine Learning')

data = pd.read_csv('data\data_clearn.csv')

book_title = st.text_input('Hãy nhập tên cuốn sách:', 'Nhà giả kim')
# Button to trigger recommendation
if st.button('Show Recommendation'):
    recommended_books = recommend_products(book_title, data)
    if recommended_books:
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, book in enumerate(recommended_books[:5], 1):
            with locals()[f'col{i}']:
                st.text(book['title'])
                st.image(book['cover_link'], use_column_width=True)
    else:
        st.write("Không tìm thấy cuốn sách hợp lệ")






