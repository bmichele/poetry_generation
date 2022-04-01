# How to Contribute

This is a small and simple contribution guide, every contribution is welcome if it is compliant with this guide.

## Code Format

Before committing new code, run [black](https://black.readthedocs.io/en/stable/#) on the modified files to format the code.

## How to Add a Model

### Adding the Code in the Generator Implementation

The code has been structured so that the `PoemGenerator` api does not change when models are added, updated or swapped.  
In this implementation there are two types of models:
 * "first_line" models: they are used to generate the first line of the poem.
 * "next_line" models: used to generate additional lines.

When adding a new model, a new file must be created in `poem_generator/generators/first_line/<model_name>.py` or `poem_generator/generators/next_line/<model_name>.py` (depending on the type).
The file must contain definitions for two functions:
 * `get_tokenizer_and_model()`, returning the tokenizer and the model used for the generation;
 * `generate()`, returning a `from poem_generator.io.candidates.PoemLineList` object containing the generated candidates

Moreover, `poem_generator.generator.PoemGenerator` must be modified so that the new implementation is used when appropriate (e.g. depending on the `PoemGenerator` configuration).  
If in doubt, you can use as a template the implementation of `poem_generator.generators.next_line.mbart_fi_single_line`.

### Adding Model Weights

We store models in Model in [huggingface model hub](https://huggingface.co/models).
When adding a new model, you should upload your model there and modify the `models/get_models.sh` to allow users
downloading the weights of the new model. 
