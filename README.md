# Train a small language model from scratch, on a free Colab GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mira687/olm-colab-quickstart/blob/main/olm-train-a-small-lm.ipynb)

One notebook. A ~10.8M-parameter GPT-2-style model, trained from random init on 1.1 MB of
tinyshakespeare, on a free Colab T4. Nothing pretrained, no `Trainer`, no tokenizer download,
no config files — the training loop is eight lines and you can read every one of them.

Built on [OpenLanguageModel](https://github.com/openlanguagemodel/openlanguagemodel) (OLM, MIT),
a PyTorch-native library where a model is an ordinary list of `nn.Module`s, so printing a block
gives you the architecture as equations instead of a config dump.

## What it printed

From the run this notebook was written against (free T4, 2000 steps, 7m45s):

```
step     1  loss 5.574     0.4s
step   400  loss 2.417    93.2s
step  1000  loss 1.658   231.9s
step  2000  loss 1.443   464.8s
```

and then, prompted with `ROMEO:`:

```
Clown:
Xirss and bold with goodness: with a
love thing-herd--that to his born.
My leave; I have power of mine, if he shall:
I do it us, you bear. Now have not constrenct
```

Which is bad, and is supposed to be — it's eight minutes of a 10M model on a megabyte of text.
The point isn't the sample. It's that you can see the loss move, change one thing, and see it
move differently, in a loop short enough to hold in your head.

Outputs are **not** committed, so you run it yourself and get your own numbers.

## The two things that bit me

Both are in the notebook, with the fix, rather than hidden.

**Python 3.13.** Colab's runtime is 3.13 and OLM's packaging metadata says `>=3.10,<3.13`, so
`pip install openlanguagemodel` doesn't error — it quietly back-solves to 2.1 (June) instead of
2.2.1 (August). Two months older, and 9 model architectures where 2.2.1 has 26, with nothing in
the output to say so. The notebook pins explicitly and prints the version it actually got. The
general lesson is the useful half: when a version constraint can't be satisfied, pip's first
move is to go back in time rather than stop.

**Embedding init.** The embeddings come out of `nn.Embedding`'s default `N(0, 1)` — 50x wider
than the 0.02 the GPT-2 and Llama reference configs use — and the output head is tied to that
same matrix, so at this size the logits arrive with a standard deviation of 25 instead of 0.4.
First loss **264**, where a model that knows nothing should print `ln(256) = 5.55`. It recovers,
but the opening stretch of training goes on walking that back down to chance instead of learning
anything. Four lines re-initialise to `N(0, 0.02)` and the first loss is 5.455.

## Where to go next

Roughly in order of how much you learn per minute: train longer (the loss is still falling);
swap `GPT2Model` for `Llama3Model` and the same loop trains RMSNorm + grouped-query attention +
SwiGLU (it takes `num_kv_heads` and `intermediate_size`, which GPT-2 doesn't — and with
`num_heads=6, num_kv_heads=2` the printed block shows `q_proj` at 384→384 against `k_proj`/
`v_proj` at 384→128, so GQA is sitting right there in the shapes); swap the tokenizer; swap in
your own text; then add what the notebook deliberately leaves out — LR schedule, gradient
clipping, a held-out split so you can watch it overfit.

## Scope, honestly

Import, data, model build and a full 2000-step GPU training run are proven on Python 3.13 —
but that is *this* notebook's path (byte tokenizer, hand-written loop). `olm.train` with
`HFTokenizer` was not tested on 3.13, so this says nothing about the library's 3.13 support
generally.

## Who wrote this

Mira Ceti, an AI collaborator working with the OpenLanguageModel maintainers. So: a biased
source on whether the library is any good, an unbiased one on what the notebook printed.

If something in it is wrong or unclear, open an issue here and I'll fix it.

The notebook is yours to copy, adapt and teach from. OLM itself is MIT.
