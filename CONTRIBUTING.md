# Contributing to try-self

Thanks for your interest in improving [**try-self**](https://github.com/DelugePrefect/try-self)! 🎉

## How to contribute

1. **Open an issue first** to discuss the change you'd like to make.
2. **Fork** the repository and create a feature branch: `git checkout -b feat/my-feature`
3. **Set up the environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Make your changes** and add tests under `tests/`.
5. **Run the checks:**
   ```bash
   python -m src.train      # ensure training still works
   pytest -q                # all tests must pass
   ```
6. **Open a pull request** with a clear description of what and why.

## Good first contributions

- Add a Dockerfile (see the roadmap in the README)
- Improve the Streamlit UI
- Add counterfactual explanations
- Expand the test suite

## Code style

- Follow PEP 8; keep functions small and typed where practical.
- Docstrings for every public function.

## Reporting bugs

Open an issue with steps to reproduce, expected vs. actual behaviour, and your
Python version / OS.
