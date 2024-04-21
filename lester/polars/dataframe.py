import json


# This only supports a subset of simple expressions for now, would be easy to extned
def _expression_to_sql_predicate(expression_tree):
    column_name = expression_tree['BinaryExpr']['left']['Column']
    operator = expression_tree['BinaryExpr']['op']
    right_literal_node = expression_tree['BinaryExpr']['right']['Literal']

    if 'String' in right_literal_node:
        right_literal = f"'{right_literal_node['String']}'"
    else:
        right_literal = list(expression_tree['BinaryExpr']['right']['Literal'].values())[0]

    operator_to_sql = {
        'Eq': '=',
        'Gt': '>',
        'GtEq': '>=',
    }

    sql_predicate = f'{column_name} {operator_to_sql[operator]} {right_literal}'

    return sql_predicate


class PolarsDuckframe:

    def __init__(self, duckframe):
        self.duckframe = duckframe

    def drop_nulls(self):
        result_duckframe = self.duckframe.dropna()
        return PolarsDuckframe(result_duckframe)

    def filter(self, expression):
        expression_tree = json.loads(expression.meta.serialize())
        predicate = _expression_to_sql_predicate(expression_tree)
        result_duckframe = self.duckframe.select(predicate)
        return PolarsDuckframe(result_duckframe)

    def join(self, other, on):
        result_duckframe = self.duckframe.join(other.duckframe, on)
        return PolarsDuckframe(result_duckframe)

    def with_column(self, expression):
        expression_tree = json.loads(expression.meta.serialize())
        new_column = expression_tree['Alias'][1]
        sql_expression = _expression_to_sql_predicate(expression_tree['Alias'][0])
        result_duckframe = self.duckframe.extended_project(new_column, sql_expression)
        return PolarsDuckframe(result_duckframe)

    def __getitem__(self, columns):
        result_duckframe = self.duckframe.project(columns)
        return PolarsDuckframe(result_duckframe)
