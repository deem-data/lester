import ast
from langchain_core.prompts import PromptTemplate
from lester.rewrite.prompts import DATAPREP_COT, FIX, FEATURISATION_COT, MODEL_COT


def extract_code(response):
    generated_code = response.content
    if '```python' in generated_code:
        generated_code = generated_code.split('```python')[1].split('```')[0]
    try:
        ast.parse(generated_code)
        return generated_code
    except Exception as e:
        print(f"SYNTACTICALLY INCORRECT CODE GENERATED:\n\n{e}\\n\n{generated_code}")


def generate_dataprep_code(task, model):
    if len(task.input_schemas()) == 1:
        hint = f"The schema of the input data for the code is: {','.join(task.input_schemas()[0])}."
    else:
        hint = f"The code consumes {len(task.input_schemas())} types of input data with the following schemas: "
        for schema in task.input_schemas():
            hint += f"{','.join(schema)}. "

    params = {
        'input_args': ', '.join(task.input_arg_names()),
        'input_hint': hint,
        'columns': task.output_columns(),
        'code': task.original_code
    }

    prompt_template = PromptTemplate.from_template(DATAPREP_COT)
    prompt = prompt_template.invoke(params)
    response = model.invoke(prompt)
    generated_code = extract_code(response)

    return generated_code


def generate_featurisation_code(task, model):
    params = {
        'columns': ', '.join(task.input_schema),
        'code': task.original_code
    }

    prompt_template = PromptTemplate.from_template(FEATURISATION_COT)
    prompt = prompt_template.invoke(params)
    response = model.invoke(prompt)
    generated_code = extract_code(response)

    return generated_code


def generate_model_code(task, model):
    params = {
        'code': task.original_code
    }

    prompt_template = PromptTemplate.from_template(MODEL_COT)
    prompt = prompt_template.invoke(params)
    response = model.invoke(prompt)
    generated_code = extract_code(response)

    return generated_code


def try_to_run(model, task, generated_code, previous_error=None, attempts=0):

    if attempts > 1:
        print("Giving up... Here is how far we got:\n")
        print(generated_code)

    else:
        if previous_error is not None:
            params = {
                'generated_code': generated_code,
                'error_message': previous_error,
            }
            prompt_template = PromptTemplate.from_template(FIX)
            prompt = prompt_template.invoke(params)
            response = model.invoke(prompt)
            generated_code = extract_code(response)

        try:
            task.evaluate_transformed_code(generated_code)
            return "ALL GOOD!!!"
        except Exception as e:
            error = repr(e)
            print(f"ERROR: {e}, trying fix")
            print(generated_code)
            print("\n\n")
            try_to_run(model, task, generated_code, error, attempts+1)
