*This project has been created as part of the 42 curriculum by mbuchet.*

# Call Me Maybe

## Description

Call Me Maybe is an introduction to **function calling** with a small language
model. Given a natural language request such as *"What is the sum of 2 and
3?"*, the program does not answer it: it chooses the function to call and its
arguments, with the right types:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Small models like **Qwen/Qwen3-0.6B** rarely produce valid JSON on their own.
This project uses **constrained decoding**: at every step, the tokens that
would break the JSON structure or the expected types are forbidden, so the
output is always valid and always matches the function definitions.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
make install        # or: uv sync
```

The model is downloaded from Hugging Face the first time the program runs.

### Usage

```bash
make run            # or: uv run python -m src
```

By default the program reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calls.json`. Every path can be changed:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

### Other Makefile rules

| Rule | Description |
|---|---|
| `make debug` | run the program with `pdb` |
| `make lint` | run `flake8` and `mypy` with the flags required by the subject |
| `make lint-strict` | run `flake8` and `mypy --strict` |
| `make clean` | remove `__pycache__`, `.mypy_cache` and `.pytest_cache` |

## Example usage

Input prompt:

```json
[{"prompt": "Greet shrek"}]
```

Command and output:

```
$ uv run python -m src
[1/1] fn_greet <- Greet shrek
```

```json
[
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {
            "name": "shrek"
        }
    }
]
```

## Algorithm explanation

The model only does one thing: given a list of tokens, it returns a score
(logit) for every possible next token. Constrained decoding works on these
scores.

1. **Vocabulary.** The text of every token is computed once from the
   vocabulary file, so the program knows which tokens are digits, quotes, etc.
2. **Prompt.** The model reads the list of functions (name, typed parameters,
   description), the question, then the start of the answer: `{"name": "`.
3. **Picking one token** (`pick_token`). The scores of all the forbidden
   tokens are set to minus infinity, then the best remaining token is taken.
   The model always chooses what it prefers, but only among allowed tokens.
4. **Function name** (`pick_word`). Only the tokens that keep the text on the
   way to an existing function name are allowed, until a full name is written.
   As soon as only one name is still possible, the rest of it is written
   directly, without asking the model.
5. **Arguments.** The program writes the fixed JSON parts itself
   (`", "parameters": {"a":`), and the model only fills the values:
   - `number` / `integer`: the number starts with `" "` or `" -"` (this is how
     Qwen writes `42` and `-42` after a colon), then only digits and a single
     `.` are allowed; `,` or `}` can end the number only when it is complete.
     A `number` is always converted to a float (`2` becomes `2.0`).
   - `string`: any token without a quote or a new line, until a token
     containing `"` ends the string. An escaped quote (`\"`) does not end it.
     JSON escapes are then decoded and the leading space is removed.
   - `boolean`: only `true` or `false`.
6. The values are converted to Python types and stored in a `FunctionCall`.

Because the structure is written by the program and every value is
constrained, the output file is always valid JSON with exactly the keys
`prompt`, `name` and `parameters`.

## Design decisions

- **One class per responsibility.** `models.py` holds the pydantic data
  classes, `utils.py` reads and writes the files, `decoder.py` contains the
  `Decoder` class that does the constrained decoding.
- **Pydantic everywhere.** All classes use pydantic, as required. The input
  files are validated when they are loaded, so a wrong type is reported with
  a clear message instead of crashing later.
- **The program writes the JSON, the model fills the gaps.** This keeps the
  output valid by construction and avoids wasting model calls on fixed text.
- **Precomputed token lists.** Digits, quote tokens and text tokens are found
  once when the decoder is created.
- **Vocabulary cache.** The vocabulary is cached per vocabulary file, so it is
  computed once even if several decoders are created.
- **`number` is always a float**, following the examples of the subject.
- **Greedy choice.** The best allowed token is always taken (no random
  sampling), so the program gives the same answer on every run.

## Performance analysis

Measured on CPU, graded with the provided moulinette:

| Test set | Score | Time (model loading included) |
|---|---|---|
| Public set | 11 / 11 (100%) | about 2 min |
| Private set | 11 / 11 (100%) | about 3 min |
| Extra tests (22 prompts, 11 new functions) | 21 / 22 | about 5 min |

- **Valid JSON:** 100% of the outputs, guaranteed by construction.
- **Reliability:** the choice is greedy, so several runs give the same
  output.
- **Speed:** almost all the time is spent in the model itself (about 1
  second per generated token on CPU). Writing the fixed JSON parts without
  the model, completing a function name as soon as only one is possible and
  stopping strings early keep the number of model calls low.

## Challenges faced

- **Tokens are not characters.** A token can contain several characters, like
  `",` or ` the`. The vocabulary had to be decoded to know what each token
  really writes.
- **Strings that never ended.** The model often prefers tokens like `",`
  instead of a single `"`, so strings kept growing. Any token containing a
  quote now ends the string.
- **Escaped characters.** Inside JSON the model writes `\\d` for `\d`; the
  strings are decoded with `json.loads` at the end.
- **Quotes inside a string.** In `Say "hello" to {name}` the model writes
  `\"`, which first ended the string. A quote right after a backslash is now
  kept in the string.
- **Negative numbers lost their sign.** Qwen writes `-12` after a colon as
  the token `" -"` (space + minus). Since the program already wrote the
  space, the model could not place its minus sign. The key is now written
  without the space and a number starts with `" "` or `" -"`.
- **A leading space in strings.** For paths like `/home/user/data.json` the
  model picks the token `" /"`; the leading space is removed at the end.
- **The model sometimes applies the function.** For "Convert 'Café au lait'
  to upper case" it may write `CAFÉ AU LAIT` instead of copying the text.
  The prompt asks to copy values exactly, which fixes most cases, but this
  remains a limit of a 0.6B model.
- **Speed.** The SDK has no cache, so each token re-reads the whole prompt.
  The prompt is kept short and fixed text is added without calling the model.

## Testing strategy

- The provided moulinette is used on both sets:

  ```bash
  cd moulinette
  uv sync
  uv run python -m moulinette prepare_exercises --set private
  uv run python -m moulinette grade_student_answers --set private \
      --student_answer_path <path to function_calls.json>
  ```

- Extra function definitions and prompts were written to test other cases:
  `integer` and `boolean` parameters, functions without parameters, several
  string parameters, negative numbers, decimals, large numbers, numbers
  written in words, quotes inside strings, accents and empty strings.
- Broken inputs were tested: missing files, invalid JSON, a JSON object
  instead of a list, wrong types, unknown parameter types, an output path
  that cannot be written. The program prints a clear error and exits with
  code 1 instead of crashing.
- `make lint` and `make lint-strict` pass without errors.

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte-level BPE tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702)
  (Willard & Louf, 2023), the idea behind constrained decoding
- [JSON specification](https://www.json.org/json-en.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [uv documentation](https://docs.astral.sh/uv/)

### Use of AI

An AI assistant (Claude) was used during this project:

- to explain the concepts of the subject (tokens, logits, constrained
  decoding) and the provided SDK;
- to review the code, find bugs and suggest simpler ways to write it;
- to help write test cases and analyse the results;
- to draft this README.

Every part of the code was read, tested and understood before being kept.
