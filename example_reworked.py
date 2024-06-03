from lester.runner import run_pipeline
from lester.ivm.feature_deletion import delete_features

from lester.examples.highlyrated_books.reworked import (label_books, random_split, encode_books, encode_target,
                                                        logistic_regression)

import time


pipeline_name = 'example_reworked'

pipeline_start_time = time.time()
run_id = run_pipeline(
    name=pipeline_name,
    source_paths={
        'books': 'data/books.csv',
        'categories': 'data/categories.csv',
        'book_tags': 'data/book_tags.csv'
    },
)
pipeline_duration = time.time() - pipeline_start_time
print(f"Pipeline took {int(pipeline_duration * 1000)}ms")


print('\n\nUpdate to delete features')
update_start_time = time.time()
delete_features(
    pipeline_name=pipeline_name,
    run_id=run_id,
    source_name='books',
    source_column_name='title',
    primary_keys=[13, 87, 113],
)
update_duration = time.time() - update_start_time
print(f"Update took {int(update_duration * 1000)}ms")
