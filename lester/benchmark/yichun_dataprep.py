# Based on https://github.com/YichunAstrid/e-commerce-use-case/tree/main/1116LogisticRegression
from lester.benchmark import DataprepCodeTransformationTask


class YichunDataprepTask(DataprepCodeTransformationTask):

    @property
    def original_code(self):
        return '''
def read_dataset(products_path, reviews_path, id):
    """
    read data
    """
    product_data = []
    with open(products_path.format(id), "r", encoding="utf-8") as product_file:
        for line in product_file.readlines():
            line = line.strip("\n")
            product_data.append(line.split("\t"))

    review_data = []
    with open(reviews_path.format(id), "r", encoding="utf-8") as review_file:
        for line in review_file.readlines():
            line = line.strip("\n")
            review_data.append(line.split("\t"))

    return product_data, review_data

def union_dataset(products_data, reviews_data):
    """
    Merge product and review data
    """
    union_data = []
    for product in products_data:
        for review in reviews_data:
            r_id = review[0]
            if r_id == product[0]:
                union_data.append([product[0], product[2], review[2], review[1], product[1]])
                if 'Sterling Silver Garnet Butterfly Earrings (1.70 CT' in product[2]:
                    print()

    return union_data

if __name__ == '__main__':
    products_path = "./data/dataset/products-data-{0}.tsv"
    reviews_path = "./data/dataset/reviews-{0}.tsv"
    products_data = []
    reviews_data = []
    for id in range(3):
        product_data, review_data = read_dataset(products_path, reviews_path, id)
        products_data += product_data
        reviews_data += review_data
    union_data = union_dataset(products_data, reviews_data)
'''

    def input_arg_names(self):
        return ['products_pathes', 'reviews_pathes']

    def input_schemas(self):
        return [['product_id', 'product_category', 'product_name'], ['product_id', 'rating', 'review']]

    def output_columns(self):
        return ['product_id', 'product_category', 'product_name', 'rating', 'review']

    def run_manually_rewritten_code(self, params):
        import lester as ld

        separator = '\t'
        product_columns = ['product_id', 'product_category', 'product_name']
        review_columns = ['product_id', 'rating', 'review']

        product_partitions = [ld.read_csv(path, header=None, sep=separator, names=product_columns)
                              for path in params['products_pathes']]
        products = ld.union(product_partitions)

        review_partitions = [ld.read_csv(path, header=None, sep=separator, names=review_columns)
                             for path in params['reviews_pathes']]
        reviews = ld.union(review_partitions)

        products_with_reviews = products.join(reviews, left_on='product_id', right_on='product_id')
        return products_with_reviews

    def evaluate_transformed_code(self, transformed_code):
        params = {
            'products_pathes': [f"data/products-data-{partition}.tsv" for partition in range(0, 3)],
            'reviews_pathes': [f"data/reviews-{partition}.tsv" for partition in range(0, 3)],
        }

        variables_for_exec = {}
        exec(transformed_code, variables_for_exec)

        generated_result = variables_for_exec['__dataprep'](products_pathes=params['products_pathes'],
                                                            reviews_pathes=params['reviews_pathes'])

        manual_result = self.run_manually_rewritten_code(params)

        data_diff = manual_result.df[self.output_columns()].compare(generated_result.df[self.output_columns()])
        assert len(data_diff) == 0
