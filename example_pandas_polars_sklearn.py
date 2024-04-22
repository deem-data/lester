from lester.runner import run_pipeline

from lester.examples.highlyrated_books.prepare_with_pandas import label_books
# from lester.examples.highlyrated_books.prepare_with_polars import label_books

from lester.examples.highlyrated_books.learn_with_sklearn import (random_split, encode_books, encode_target,
                                                                  logreg_with_hpo)

run_pipeline(
    source_paths={'books': 'data/books.csv', 'categories': 'data/categories.csv', 'book_tags': 'data/book_tags.csv'},
)
