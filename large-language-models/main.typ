#import "template.typ": *
#import "@preview/diagraph:0.2.1": *

#show: project.with(
  title: "Large Language Models",
  authors: (
    (name: "Thomas Gassmann", email: "tgassmann@student.ethz.ch"),
  ),
  date: "Feb 18, 2025",
)

#let define = $attach(=, t: "def")$
#let EOS = $#math.script("EOS")$


= Large Language Models

== Probabilistic Foundations

A *language model* is a function $p(underbrace(dot, "symbol/token") | underbrace(dot, "prompt"))$. An *alphabet* $Sigma$ is a finite, non-empty set. A *string* is a finite sequence of symbols drawn from an alphabet. $Sigma^ast$ is the set of all strings over $Sigma$, and is countable. $Sigma^infinity$ is the set of infinite sequences over $Sigma$. A *language* is a subset of $Sigma^ast$ for some alphabet $Sigma$.

#colorbox(title: [Language Model (Informal)], color: silver)[
  Given an alphabet $Sigma$ and a distinguished $EOS in.not Sigma$, a language model is a collection of conditional probability distributions $p(y | bold(y))$ (probability of $y$ occuring as next token after string $bold(y)$) for $y in Sigma union {EOS}$ and $bold(y) in Sigma^ast$.
]

An *energy function* is a function $hat(p) : Sigma^ast arrow RR$.

#colorbox(title: [Globally normalized model])[
  Let $hat(p)_("GN")(bold(y)) : Sigma^ast arrow RR$ be an energy function. A globally normalized model (GNM) is defined as:
  $
    p_("LM")(bold(y)) define (exp(- hat(p)_("GN")(bold(y)))) / (sum_(y' in Sigma^ast) exp(-hat(p)_("GN")(bold(y')))) define exp(-hat(p)_"GN" (bold(y)))
  $
  where $Z_"G" define sum_(bold(y') in Sigma^ast) exp(-hat(p)_"GN" (bold(y')))$ is the normalization constant.
]

Any normalizable energy function $hat(p)_"GN"$ (meaning $Z_G$ is finite) induces a language model, i.e., a distribution over $Sigma^ast$.

#colorbox(title: [Sequence model], color: silver)[
  For an alphabet $Sigma$ a sequence model is defined as a set of conditional probability distributions $p_"SM"(y | bold(y))$ for $y in Sigma, bold(y) in Sigma^ast$. $bold(y)$ is called history/context.
]

#colorbox(title: [Locally normalized model /autoregressive model])[
  For $p_"SM"$ a sequence model over $overline(Sigma)$: A locally normalized language model (LNM) over $Sigma$ is defined as:
  $
    p_"LN" (bold(y)) define p_"SM" (EOS | bold(y)) product_(t=1)^T p_"SM" (y_t |  bold(y)_(<t))
  $
  for $y in Sigma^ast, |bold(y)| = T$. The LNM is tight if $sum_(bold(y) in Sigma^ast) p_"LN" (bold(y)) = 1$.
]

#colorbox(title: [Prefix probability], color: silver)[
  Let $p_"LM"$ be a language model. The prefix probability of $p_"LM"$ is:
  $
    pi (bold(y)) define sum_(y' in Sigma^ast) p_"LM" (bold(y) bold(y'))
  $
  i.e. the cumulative probability of all strings in the language beginning with $bold(y)$.
]

Any language model can be locally normalized. 