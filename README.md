# micrograd-from-scratch

A tiny automatic differentiation engine and neural network library, built
from nothing in pure Python. No NumPy, no PyTorch — every gradient in here
is computed by code in this repository.

Built by following Andrej Karpathy's *Neural Networks: Zero to Hero*
(lecture 1), typing everything by hand rather than copying.

---

## What it does

```python
n = MLP(3, [4, 4, 1])          # 3 inputs -> 4 -> 4 -> 1 output, 41 parameters

xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(50):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
```

Loss falls from ~7 to ~0.01 over 50 steps, and the predictions converge on
the targets.

---

## What's inside

**`Value`** — a scalar that remembers where it came from. Every operation
records its parents and attaches a local backward function, so an
expression builds a computation graph as a side effect of ordinary
arithmetic.

Supported: `+`, `*`, `-`, `/`, `**`, unary negation, `tanh`, `exp`, plus
the reflected operators (`__radd__`, `__rmul__`, `__rsub__`,
`__rtruediv__`) so that `2 * a` and `2 - a` work as well as `a * 2`.

**`backward()`** — topologically sorts the graph, then walks it in reverse
calling each node's local backward function. One call fills in every
gradient in the expression.

**`draw_dot()`** — renders the graph with graphviz, showing data and
gradient on every node. (Copied from the lecture; it's a visualization
tool, not part of the engine.)

**`Neuron` / `Layer` / `MLP`** — a neuron holds weights and a bias, a layer
holds neurons, an MLP holds layers. `parameters()` flattens the whole
network into one list for the optimizer to iterate over.

Subtraction and division are built from `__neg__` and `__pow__` rather than
getting their own backward functions — five operators, no extra gradient
code.

---

## What I learned (including the bugs)

**Reaching into `.data` silently breaks the graph.** My first `__sub__`
was `self + (-(other.data))`. The arithmetic was correct and no error was
raised — but pulling the raw number out meant `other` was never recorded as
a parent, so its gradient stayed at 0.0 forever. In a real network that
weight would simply never learn, and nothing would ever complain. Fixed by
keeping everything as `Value` objects: `self + (-other)`.

**Gradients accumulate, so they have to be zeroed.** Using `=` instead of
`+=` inside a backward function overwrites earlier contributions — visible
immediately with `b = a + a`, where `a.grad` came out 1.0 instead of 2.0.
But `+=` means gradients also pile up across separate backward passes,
which is why every training loop starts by resetting them. This is exactly
what `optimizer.zero_grad()` does in PyTorch.

**A learning rate of 1.0 blows the weights up.** Steps that large overshoot
badly; the weights grew until `math.exp(2*x)` overflowed a float and `tanh`
raised `OverflowError`.

**Saturated tanh kills learning.** After the blowup, every neuron output sat
at −0.99999995, out in the flat tail where the local derivative `1 - t²` is
about 10⁻⁸. Gradients were multiplied by near-zero on the way back, so the
loss dropped by roughly 10⁻¹⁰ per step and the network was effectively
frozen at loss 8.0. This is the vanishing gradient problem, met firsthand.

**Notebook state is invisible and lies to you.** Most of the time I lost on
this project went to cells run out of order — stale objects, gradients
wiped by a re-run graph cell, a snapshot drawn before the values it was
meant to show. A notebook that can't survive *Restart and run all* is
broken even when it looks fine.

---

## Running it

Open `01_micrograd.ipynb` in Colab or Jupyter. Needs `graphviz` for the
diagrams:

```
pip install graphviz
apt-get install graphviz
```

---

## Credit

Following [Andrej Karpathy's micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0).
The engine is reimplemented by hand; `draw_dot` is taken from the lecture
notebook.
