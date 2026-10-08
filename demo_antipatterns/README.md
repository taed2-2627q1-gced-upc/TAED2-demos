# Pytest antipatterns

Runnable examples of what goes wrong WITHOUT fixtures and parametrization. They are outside `testpaths`, so they never
run with the regular suite. Run one explicitly, e.g.:

```bash
uv run pytest demo_antipatterns/test_no_fixture.py --no-cov --durations=0
```
