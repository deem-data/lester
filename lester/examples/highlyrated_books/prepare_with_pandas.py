from lester.context import datasource, prepare, DataframeDialect


@prepare(
    dialect=DataframeDialect.PANDAS,
    sources=[
        datasource('books', track_provenance_by=['goodreads_book_id']),
        datasource('categories', track_provenance_by=['tag_id']),
        datasource('book_tags')])
def label_books(books, categories, book_tags):

    english_books = books \
        .dropna() \
        .query("language_code == 'eng'")

    popular_categories = categories.query("popularity >= 10")
    categories_with_books = popular_categories.merge(book_tags, on='tag_id')

    labeled_books = english_books.merge(categories_with_books, on='goodreads_book_id')
    labeled_books['is_highly_rated'] = labeled_books.eval('average_rating > 4.2')

    return labeled_books
