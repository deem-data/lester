import polars as pl
from lester.context import datasource, prepare, DataframeDialect


@prepare(
    dialect=DataframeDialect.POLARS,
    sources=[
        datasource('books', track_provenance_by=['goodreads_book_id']),
        datasource('categories', track_provenance_by=['tag_id']),
        datasource('book_tags')])
def label_books(books, categories, book_tags):

    english_books = books \
        .drop_nulls() \
        .filter(pl.col("language_code") == 'eng')

    popular_categories = categories.filter(pl.col("popularity") >= 10)
    categories_with_books = popular_categories.join(book_tags, on='tag_id')

    labeled_books = english_books.join(categories_with_books, on='goodreads_book_id')
    labeled_books = labeled_books.with_column((pl.col("average_rating") > 4.2).alias("is_highly_rated"))

    return labeled_books
