from lester.runner import run_pipeline

from lester.examples.highlyrated_books.prepare_with_pyspark import label_books
from lester.examples.highlyrated_books.encode_with_sparkml import random_split, encode_books
from lester.examples.highlyrated_books.learn_with_sparkml import logreg_with_fixed_hyperparams

run_pipeline(
    name='example_pyspark_sparkml',
    source_paths={'books': 'data/books.csv', 'categories': 'data/categories.csv', 'book_tags': 'data/book_tags.csv'},
)
