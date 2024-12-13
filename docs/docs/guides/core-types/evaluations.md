import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Evaluations

To systematically improve your LLM application, it's helpful to test your changes against a consistent dataset of potential inputs so that you can catch regressions and inspect your application's behaviour under different conditions. In Weave, the `Evaluation` class is designed to assess the performance of a `Model` on a test dataset.

In a Weave evaluation, a set of examples is passed through your application, and the output is scored according to multiple scoring functions. The result provides you with an overview of your application's performance in a rich UI to summarizing individual outputs and scores.

This page describes the steps required to [create an evaluation](#create-an-evaluation). A complete [Python example](#python-example) and additional [usage notes and tips](#usage-notes-and-tips) are also included.

![Evals hero](../../../static/img/evals-hero.png)

## Create an evaluation

To create an evaluation in Weave, follow these steps:

1. [Define an evaluation dataset](#define-an-evaluation-dataset)
2. [Define scoring functions](#define-scoring-functions)
3. [Define an evaluation target](#define-an-evaluation-target)

### Define an evaluation dataset

First, create a test dataset to evaluate your application against. Generally, the dataset should include failure cases that you want to test for, similar to software unit tests in Test-Driven Development (TDD). You have two options to create a dataset:

1. Define a [Dataset](/guides/core-types/datasets).
2. Define a list of dictionaries with a collection of examples to be evaluated. For example:
   ```python
   examples = [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "Who wrote 'To Kill a Mockingbird'?", "expected": "Harper Lee"},
        {"question": "What is the square root of 64?", "expected": "8"},
    ]
   ```
   

Next, [define scoring functions](#define-scoring-functions).

### Define scoring functions

Next, create a list of scoring functions, also known as _scorers_. Scorers are used to score the performance of an AI system against the evaluation dataset. Scorers must have a `model_output` keyword argument. Other arguments are user defined and are taken from the dataset examples. The scorer will only use the necessary keys by using a dictionary key based on the argument name.

The options available depend on whether you are using Typescript or Python:

<Tabs groupId="programming-language">
  <TabItem value="python" label="Python" default>
  There are three types of scorers available for Python:

    :::tip
    [Predefined scorers](../evaluation/predefined-scorers.md) are available for many common use cases. Before creating a custom scorer, check if one of the predefined scorers can address your use case.
    :::

    1. [Predefined scorer](../evaluation/predefined-scorers.md): Pre-built scorers designed for common use cases.
    2. [Function-based scorers](../evaluation/custom-scorers#function-based-scorers): Simple Python functions decorated with `@weave.op`.
    3. [Class-based scorers](../evaluation/custom-scorers#class-based-scorers): Python classes that inherit from `weave.Scorer` for more complex evaluations.

    In the following example, the function-based scorer `match_score1()` will take `expected` from the dictionary for scoring.

    ```python
    import weave

    # Collect your examples
    examples = [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "Who wrote 'To Kill a Mockingbird'?", "expected": "Harper Lee"},
        {"question": "What is the square root of 64?", "expected": "8"},
    ]

    # Define any custom scoring function
    @weave.op()
    def match_score1(expected: str, model_output: dict) -> dict:
        # Here is where you'd define the logic to score the model output
        return {'match': expected == model_output['generated_text']}
    ```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">
     Only [function-based scorers](../evaluation/custom-scorers#function-based-scorers) are available for Typescript. For [class-based](../evaluation/custom-scorers.md#class-based-scorers) and [predefined scorers](../evaluation/predefined-scorers.md), you will need to use Python.
  </TabItem>
</Tabs>

:::tip
Learn more about [how scorers work in evaluations and how to use them](../evaluation/scorers.md). 
:::

Next, [define an evaluation target](#define-an-evaluation-target).

### Define an evaluation target

Once your test dataset and scoring functions are defined, you can define the target for evaluation. You can [evaluate a `Model`](#evaluate-a-model) or any [function](#evaluate-a-function). 

#### Evaluate a `Model` 

`Models` are used when you have attributes that you want to experiment with and capture in Weave.
To evaluate a `Model`, use the `evaluate` method from `Evaluation`. 

The following example runs `predict()` on each example and scores the output with each scoring function defined in the `scorers` list using the `examples` dataset.

```python
from weave import Model, Evaluation
import asyncio

class MyModel(Model):
    prompt: str

    @weave.op()
    def predict(self, question: str):
        # Here's where you would add your LLM call and return the output
        return {'generated_text': 'Hello, ' + self.prompt}

model = MyModel(prompt='World')

evaluation = Evaluation(
    dataset=examples, scorers=[match_score1]
)
weave.init('intro-example') # begin tracking results with weave
asyncio.run(evaluation.evaluate(model))
```

#### Evaluate a function 

Alternatively, you can also evaluate any function by wrapping it with a `@weave.op()`.

```python
@weave.op()
def function_to_evaluate(question: str):
    # here's where you would add your LLM call and return the output
    return  {'generated_text': 'some response'}

asyncio.run(evaluation.evaluate(function_to_evaluate))
```

## Python example

The following example shows an evaluation that uses `dataset` and two scorers, `match_score1` and `match_score2`, to run evaluations on `model` and `function_to_evaluate`. You can use this example as a template for your own evaluations.

```python
from weave import Evaluation, Model
import weave
import asyncio

examples = [
    {"question": "What is the capital of France?", "expected": "Paris"},
    {"question": "Who wrote 'To Kill a Mockingbird'?", "expected": "Harper Lee"},
    {"question": "What is the square root of 64?", "expected": "8"},
]

@weave.op()
def match_score1(expected: str, model_output: dict) -> dict:
    return {'match': expected == model_output['generated_text']}

@weave.op()
def match_score2(expected: dict, model_output: dict) -> dict:
    return {'match': expected == model_output['generated_text']}

class MyModel(Model):
    prompt: str

    @weave.op()
    def predict(self, question: str):
        # here's where you would add your LLM call and return the output
        return {'generated_text': 'Hello, ' + question + self.prompt}

model = MyModel(prompt='World')
evaluation = Evaluation(dataset=examples, scorers=[match_score1, match_score2])

# Start tracking the evaluation
weave.init('intro-example')

asyncio.run(evaluation.evaluate(model))

@weave.op()
def function_to_evaluate(question: str):
    # here's where you would add your LLM call and return the output
    return  {'generated_text': 'some response' + question}

asyncio.run(evaluation.evaluate(function_to_evaluate))
```

## Usage notes and tips

### Change the name of an evaluation

You can change the name of the evaluation by passing a `name` parameter to the `Evaluation` class.

```python
evaluation = Evaluation(
    dataset=examples, scorers=[match_score1], name="My Evaluation"
)
```

You can also change the name of individual evaluations by setting the `display_name` key of the `__weave` dictionary. Using the `__weave` dictionary sets the call display name which is distinct from the `Evaluation` object name. In the UI, you will see the display name if set. Otherwise, the `Evaluation` object name will be used.

```python
evaluation = Evaluation(
    dataset=examples, scorers=[match_score1]
)
evaluation.evaluate(model, __weave={"display_name": "My Evaluation Run"})
```