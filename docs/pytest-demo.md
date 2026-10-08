# Pytest demo <!-- omit in toc -->
In this demo we will see the main features of [Pytest](https://docs.pytest.org/en/latest/) to test a simple machine
learning project.

## Contents <!-- omit in toc -->
- [Install Pytest](#install-pytest)
- [Unit tests](#unit-tests)
- [Why fixtures and parametrization?](#why-fixtures-and-parametrization)
- [Model behavioral tests](#model-behavioral-tests)
- [Configure Pytest](#configure-pytest)
- [Run the tests](#run-the-tests)
- [Measure test coverage (Optional)](#measure-test-coverage-optional)

## Install Pytest
The first step is to install `pytest`. To do this, we can run the
following command:
```bash
uv add --group dev pytest
```

## Unit tests
A unit test checks one small function in isolation, with hand-built inputs and no model, disk or network. The
functions in [`preprocess.py`](../src/data/preprocess.py) are a good fit. See
[`test_preprocess.py`](../tests/test_preprocess.py):

- the `sample_df` fixture builds a tiny DataFrame (with nulls, empty strings and duplicates) for `remove_empty_or_duplicate`;
- `test_sanitize_text` is parametrized over several raw/expected pairs, one per cleaning rule.

One case, `multiple-gaps`, **fails on purpose**: `sanitize_text` uses `str.replace`, which only replaces the **first**
match, so `"a  b\nc  d"` is not fully normalized. This is a real bug that the unit test found. Using `replace_all`
instead fixes it.

## Why fixtures and parametrization?
Before using them, see what goes wrong without them. The runnable examples live in
[`demo_antipatterns/`](../demo_antipatterns), outside `testpaths`, so they never run with the regular suite:

```bash
uv run pytest demo_antipatterns/test_no_fixture.py --no-cov --durations=0
```

| Without... | File | What goes wrong |
|---|---|---|
| a fixture | [`test_no_fixture.py`](../demo_antipatterns/test_no_fixture.py) | Setup is copy-pasted into every test and the model is reloaded each time. The first call took 1.25 s and the next two about 0.15 s, because Hugging Face caches the weights; on a cold cache or a larger model the gap is much bigger. A fixture with `scope="session"` loads it once. |
| test isolation | [`test_shared_state.py`](../demo_antipatterns/test_shared_state.py) | Tests share a module-level list. `test_only_one_result_recorded` passes alone but fails after the other test, so results depend on execution order. A function-scoped fixture gives every test a fresh object. |
| `parametrize` | [`test_no_parametrize.py`](../demo_antipatterns/test_no_parametrize.py) | Several checks live in one test. The first failing assert stops the function, so the cases after it are never run and you see only one failure. `parametrize` runs and reports each case separately. |
| `ids` | [`test_bad_ids.py`](../demo_antipatterns/test_bad_ids.py) | With object parameters the failure is reported as `test_labels[case1]`. With `pytest.param(..., id="negation")` it reads `test_mft_simple_sentences[negation]`. |

Fixtures that need cleanup can also `yield`: the code after the `yield` runs even if the test fails (as in the `client`
fixture in [`test_api.py`](../tests/test_api.py)).

Shared fixtures live in [`conftest.py`](../tests/conftest.py), which pytest loads automatically:
- `pipe` loads the model once per session;
- `positive_score` is a *factory fixture*: it returns a function `text -> P(positive)`.

## Model behavioral tests
Accuracy on a test set tells us how well the model does on average, not *what* it does on specific inputs. Behavioral
tests, as proposed in [CheckList](https://aclanthology.org/2020.acl-main.442/) (Ribeiro et al., 2020), probe it with
controlled inputs. See [`test_model_behavior.py`](../tests/test_model_behavior.py):

| Type | Idea | Test |
|---|---|---|
| **MFT** (minimal functionality) | Simple, unambiguous inputs must be right. | `test_mft_simple_sentences` |
| **INV** (invariance) | A change that should not matter (name, casing, neutral sentence, typo) must not change the label. | `test_inv_label_does_not_change` |
| **DIR** (directional expectation) | A change with a known direction must move P(positive) that way. Adding a negative sentence must lower it by at least 0.1; adding a positive one must not lower it by more than 0.01. | `test_dir_score_moves_in_expected_direction` |

Two cases fail on the current model: negation (`"The movie was not bad."`) and a typo (`"I lovd this movie."` flips the
label). They are marked `xfail(strict=True)` through `pytest.param(..., marks=...)`, so behavioral tests document real
weaknesses without breaking the build. If a retrained model fixes one, `strict=True` flags it.

The accuracy test stays in [`test_model.py`](../tests/test_model.py). For more checks and a tool to generate them, see the
[CheckList library](https://github.com/marcotcr/checklist).

## Configure Pytest
Finally, we add the following lines to the [`pyproject.toml`](../pyproject.toml) file to configure `pytest`:
```toml
[tool.pytest.ini_options]
pythonpath = "."
testpaths = "tests"
```

In detail:
- `pythonpath` specifies the path to the source code;
- `testpaths` specifies the directory where the tests are located (this is why `demo_antipatterns/` is not collected);

## Run the tests
We can now run the tests using the following command:
```bash
pytest
```

## Measure test coverage (Optional)
When the source code grows it can be difficult to know what is not being tested. For such cases, we can use the `pytest-cov` package to measure the test coverage. This package will generate a coverage report that shows which parts of the code are not being tested.

To install it we can run the following command:
```bash
uv add --group dev pytest-cov
```

Then, we can add the following to our `pyproject.toml` file:

```toml
[tool.coverage.run]
omit = ["src/data/validate_data.py", "src/data/gx_context_configuration.py", "src/modeling/train.py"]

[tool.pytest.ini_options]
...
addopts = "--junitxml=out/tests-report.xml --cov=src --cov-report=html:reports/coverage"
```

In detail:
- `omit` specifies the files that should be omitted from the coverage report. In this case, we omit some scripts that only
call functions that have been defined elsewhere.

- `addopts` specifies options to pass to `pytest`. In this case, we are specifying the path to the test results file
(`out/tests-report.xml`), which can be read by continuous integration tools, the path to the module we want to measure
the coverage (`src`), and the format (HTML) and directory where the coverage report should be saved (`reports/coverage`).
