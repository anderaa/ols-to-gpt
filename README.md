# OLS to GPT

**A single continuous thread from univariate linear regression to a working GPT in 20 steps.**

[Open the notebook →](ols_to_gpt.ipynb)

There is a popular idea that artificial intelligence is something categorically different from conventional statistics and machine learning, and that at some point a model crosses a threshold and becomes something new, something that thinks. This notebook argues that idea is wrong.

It begins with the simplest possible model: a straight line fit to a handful of points via ordinary least squares (OLS). It ends with the architecture behind GPT, the class of model that powers modern large language models. Every step between them implements only minor modifications or additions. At no point is there a clean break between convential methods and AI.

Someone who understands linear regression already understands much of what makes a transformer work. The concepts are the same: a parametric model, a loss function, an optimizer that minimizes it, and a prediction on new data. The mystique surrounding modern AI is largely a product of scale and opacity, not of fundamentally new ideas. The core machinery (maximum likelihood estimation, gradient descent, and learned representations) has been in the statistical literature for decades.

The unifying goal throughout is **prediction**: given some inputs, what is the most probable output? From predicting a continuous value with least squares, to predicting the next character in a Shakespeare play with a stacked transformer, the model is always learning a mapping from inputs to outputs that generalises beyond the training data. The inputs get richer, the mappings get more complex, and the outputs get more expressive — but the logic never changes.

## Outline

| Step | Model | Target | Input | Estimator | Framework |
|------|-------|--------|-------|-----------|-----------|
| 0 | Univariate linear regression | Continuous | Continuous (p=1) | Least squares | NumPy |
| 1 | Multivariate linear regression | Continuous | Continuous (p>1) | Least squares | NumPy |
| 2 | Multivariate linear regression | Continuous | Continuous (p>1) | MLE + Newton's method | NumPy |
| 3 | Logistic regression | Binary | Continuous | MLE + Newton's method | NumPy |
| 4 | Logistic regression | Binary | Continuous | MLE + gradient descent | NumPy |
| 5 | Logistic regression | Binary | Continuous | MLE + gradient descent | PyTorch |
| 6 | Neural network — sigmoid hidden layer | Binary | Continuous | MLE + gradient descent | PyTorch |
| 7 | Neural network — ReLU hidden layer | Binary | Continuous | MLE + gradient descent | PyTorch |
| 8 | Neural network — multi-class classification | Multi-class | Continuous | MLE + gradient descent | PyTorch |
| 9 | Neural network — discrete inputs via one-hot encoding | Multi-class | Discrete (one-hot) | MLE + gradient descent | PyTorch |
| 10 | Neural network — discrete inputs via learned embeddings | Multi-class | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 11 | Character-level LM — 2-character context, concatenation | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 12 | Character-level LM — single-head self-attention | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 13 | Character-level LM — causal masking | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 14 | Character-level LM — last output only, 5-char context | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 15 | Character-level LM — positional embeddings | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 16 | Character-level LM — multi-head self-attention | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 17 | Character-level LM — residual connections + layer norm | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 18 | Small GPT — stacked transformer blocks | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 19 | Small GPT — feed-forward sublayer inside transformer block | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch |
| 20 | Scaled-up GPT — MPS GPU, Tiny Shakespeare, early stopping | Next character | Discrete (embedding) | MLE + gradient descent | PyTorch + MPS |

## Getting started

```bash
git clone https://github.com/anderaa/ols_to_gpt.git
cd ols_to_gpt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ols_to_gpt.ipynb
```

Then run the cells in order from top to bottom — each step builds on the previous one.

Step 20 trains a small GPT on [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (a copy is included as `tinyshakespeare.txt`). On an Apple Silicon Mac it uses the `mps` device automatically; on other machines it falls back to CPU.

## Repository contents

- `ols_to_gpt.ipynb` — the notebook, 20 steps from least squares to GPT
- `diagrams.py` — model architecture diagrams used throughout the notebook
- `plots.py` — shared plotting helpers (convergence curves, confusion matrices, actual-vs-predicted)
- `tinyshakespeare.txt` — training corpus for step 20
- `scatter_with_trendline.xlsx` — Excel output from step 0, illustrating that the very first model is just a trendline
- `resources/` — images used by the notebook
- `requirements.txt` — pinned Python dependencies
