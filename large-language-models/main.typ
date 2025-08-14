#import "template.typ": *
#import "@preview/xarrow:0.3.1": xarrow

#show: project.with(
  authors: (
    (name: "Thomas Gassmann", email: "tgassmann@student.ethz.ch"),
  )
)

#let define = $attach(=, t: "def")$
#let EOS = $#math.script("EOS")$

== Probability theory

Traditionally, a probability space is a triple $(Omega, cal(F), bb(P))$ where $bb(P)$ is a measure, $bb(P) [Omega] = 1$ and $cal(F) subset.eq cal(P) (Omega)$. To resolve e.g. the paradox of the infinite coin toss, we require $cal(F)$ to be a $sigma$-algebra.

#colorbox(title: [$sigma$ algebra], color: silver)[
  A set $cal(F) subset.eq cal(P) (Omega)$ is called a $sigma$-algebra s.t.:
  - $Omega in cal(F)$
  - $Sigma in cal(F) arrow.r.double Sigma^complement in cal(F)$
  - If $Sigma_1, Sigma_2, dots in cal(F)$, then $union.big_(n=1)^infinity Sigma_n in cal(F)$
]

#colorbox(title: [Probability measure], color: silver)[
  A probability $bb(P)$ over a measure space $(Omega, cal(F))$ is a function $bb(P): cal(F) arrow [0,1]$ s.t.:
  - $bb(P) (Omega) = 1$
  - If $Sigma_1, Sigma_2, dots in cal(F)$ is a countable sequence of disjoint events, then $bb(P) [union.big_(n=1)^infinity Sigma_n] = sum_(n=1)^infinity bb(P) [Sigma_n]$.

  i.e. we have a measure space if $Omega$ is a set and $cal(F)$ is a $sigma$-algebra over $Omega$.
]

#colorbox(title: [Measureable function], color: silver)[
  Let $(Omega, cal(F))$ and $(S, T)$ be measure spaces. A random variable is a measurable function from $Omega arrow S$.

  A *measurable function* $x: Omega arrow S$ is such that $x^(-1) (Sigma)$ is measurable for $Sigma$ measurable, i.e. $Sigma in T arrow.double x^(-1) (Sigma) in cal(F)$.
]

*Infinite coin toss paradox*: $1 = p(Sigma^infinity) = p(union_(omega in Sigma^infinity) {omega}) = sum_(omega in Sigma^infinity) p({omega}) = 0$.

#colorbox(title: [KL divergence], color: silver)[ For two distributions $P$ and $Q$ we define $D_"KL" (P || Q) = sum_(x in cal(X)) P(x) log(P(x) / (Q(x)))$.
]

#colorbox(title: [Cross entropy], color: silver)[
  For two distributions $P$ and $Q$ over a set $cal(X)$, we define $H(P, Q) = - sum_(x in cal(X)) P(x) log(Q(x))$. The entropy $H(P) = H(P, P)$. Note that $H(P, Q) = H(P) + D_"KL" (P || Q)$.
]

== Foundations

An *alphabet* $Sigma$ is a finite, non-empty set. A *string* is a finite sequence of symbols drawn from an alphabet. $Sigma^ast$ is the set of all strings over $Sigma$, and is countable. $Sigma^infinity$ is the set of infinite sequences over $Sigma$. A *language* is a subset of $Sigma^ast$ for some alphabet $Sigma$. A language model is a distribution over $Sigma^ast$.

// #colorbox(title: [Language Model (Informal)], color: silver)[
//   Given an alphabet $Sigma$ and a distinguished $EOS in.not Sigma$, a language model is a collection of conditional probability distributions $p(y | bold(y))$ (probability of $y$ occuring as next token after string $bold(y)$) for $y in Sigma union {EOS}$ and $bold(y) in Sigma^ast$.
// ]

// An *energy function* is a function $hat(p) : Sigma^ast arrow RR$.

#colorbox(title: [Globally normalized model])[
  Let $hat(p)_("GN")(bold(y)) : Sigma^ast arrow RR$ be an energy function. A globally normalized model (GNM) is defined as:
  $
    p_("LM")(bold(y)) define (exp(- hat(p)_("GN")(bold(y)))) / (sum_(y' in Sigma^ast) exp(-hat(p)_("GN")(bold(y')))) define exp(-hat(p)_"GN" (bold(y)))
  $
  where $Z_"G" define sum_(bold(y') in Sigma^ast) exp(-hat(p)_"GN" (bold(y')))$ is the normalization constant.
]

Any normalizable energy function $hat(p)_"GN"$ (meaning $Z_G$ is finite) induces a language model, i.e., a distribution over $Sigma^ast$.

#colorbox(title: [Sequence model], color: silver)[
  For an alphabet $Sigma$ a sequence model is defined as a set of conditional probability distributions $p_"SM"(y | bold(y))$ for $y in overline(Sigma)$ ($overline(Sigma) = Sigma union \{ EOS \}$) $bold(y) in Sigma^ast$. $bold(y)$ is called history/context. That is, we have $\{ p_"SM" (y | bold(y)) \}_(bold(y) in Sigma^ast)$ for $y in overline(Sigma)$.
]

A sequence model is a probability distribution over $Sigma^ast union Sigma^infinity$.

#colorbox(title: [Locally normalized model /autoregressive model])[
  For $p_"SM"$ a sequence model over $overline(Sigma)$: A locally normalized language model (LNM) over $Sigma$ is defined as:
  $
    p_"LN" (bold(y)) define p_"SM" (EOS | bold(y)) product_(t=1)^(|bold(y)|) p_"SM" (y_t |  bold(y)_(<t))
  $
  for $y in Sigma^ast$. The LNM is tight if $sum_(bold(y) in Sigma^ast) p_"LN" (bold(y)) = 1$.
]

#colorbox(title: [Prefix probability], color: silver)[
  Let $p_"LM"$ be a language model. The prefix probability of $p_"LM"$ is:
  $
    pi (bold(y)) define sum_(y' in Sigma^ast) p_"LM" (bold(y) bold(y'))
  $
  i.e. the cumulative probability of all strings in the language beginning with $bold(y)$.
]

Any language model can be locally normalized. TODO: should know how to prove this, telescoping product, see page 24

#colorbox(title: [Tightness conditions], color: silver)[
  1. Let $p_EOS (t) = (sum_(omega in Sigma^(t-1)) p_"LN" (omega) p_"LN" (EOS | omega))/(sum_(omega in Sigma^(t-1)) p_"LN" (omega))$. $p_"LN"$ is *tight* iff. $exists t >= 1: p_EOS (t) = 1 or sum_(t=1)^infinity p_EOS (t) = infinity$.
  2. If $p_"LN" (EOS | bold(y)) >= f(t)$ for all $bold(y) in Sigma^t$ (and $t$) and $sum_(t=1)^(infinity) f(t) = infinity$, then $p_"LN"$ is tight.
]

The *softmax* function is defined as $"softmax"(x)_i = exp(x_i / tau) / (sum_(j=1)^n exp(x_j / tau)))$ for a temperature parameter $tau > 0$. As $t arrow infinity$ the distribution becomes uniform, as $tau arrow 0$ the distribution becomes spiked. We have $"softmax"(x) = "argmax"_(p in Delta^(n-1)) p^top x - tau sum_(i=1)^n p_i log(p_i)$.

The *sparsemax* function is defined as $"sparsemax" (x) = "argmin"_(z in Delta^(n-1)) ||z - x||_2^2$. This addresses the drawback of softmax that $"softmax"_i (z) > 0 space forall z, i$ (in some tasks sparse probability is preferable).

#colorbox(title: [Representation-based Language Model (RBLM)])[
  An embedding matrix $E$ and an encoding function $"enc": Sigma^ast mapsto RR^d$ define a locally normalized language model using the sequence model:
  $
    p_"SM" (overline(y)_t | bold(overline(y))_(<t)) = "softmax" (E "enc" (bold(overline(y))_(<t)))_(overline(y)_t)
  $
]

We define $s = max_(y in Sigma) ||e(y) - e(EOS)||_2$ and $z(t) = max_(omega in Sigma^t) ||"enc"(omega)||_2$, where $e(dot)$ is the symbol embedding function.

*RBLM Tightness:* If $s dot z(t) <= log(t)$ for all $t >= N$ for some $N$, then the induced RBLM is *tight*. In particular, $"enc"(dot)$ is bounded, then the model is *tight*.

== Finite State Language Models

#colorbox(title: [FSA])[
  An FSA is a tuple $cal(A) = (Q, Sigma, delta, I, F)$, where $Q$ is a finite set of states, $Sigma$ is a alphabet, $delta subset.eq Q times (Sigma union {epsilon}) times Q$ are the transitions, and $I, F subset.eq Q$ are the initial/final states.
]

#colorbox(title: [Weighted FSA])[
  An WFSA is a tuple $cal(A) = (Q, Sigma, delta, lambda, rho)$, where $delta subset.eq Q times (Sigma union {epsilon}) times RR times Q$ are the (weighted) transitions, and $lambda, rho: Q mapsto RR$ are the initial/final weights.
]

A WFSA is *probabilistic* if $lambda, rho$ and the transition weights form are non-negative, $sum_(q in Q) lambda(q) = 1$ and for all $q in Q$ we have $rho(q) + sum_(q xarrow(a "/" w) q') w = 1$.

The *weight of a path* $pi = q_1 xarrow(a_1 "/" w_1) q_2 dot dot q_N$ in a WFSA $cal(A)$ is given by $w(pi) = lambda(q_1) product_(i=1)^(N) w_i rho(q_N)$. $Pi(cal(A), y)$ is the set of all paths where $cal(A)$ yields $y$.

// TODO: verify last below
The *Allsum* of a WFSA $cal(A)$ is defined as $Z(cal(A)) = sum_(y in Sigma^ast) cal(A) (y) = sum_(y in Sigma^ast) sum_(pi in Pi(A, y)) w(pi) = arrow(lambda) sum_(d=0)^infinity T^d arrow(rho) = arrow(lambda) (I - T)^(-1) arrow(rho)$, where $T$ is the transition matrix of $cal(A)$.

*Tightness of PFSA*: A state $q in Q$ is accessible if there exists a non-zero weighted path from an initial state to $q$. It is co-accessible if there exists a non-zero weighted path from $q$ to a final state. A PFSA is *tight* iff. all accessible states are co-accessible.

// TODO: CFG skipped

== RNNs

A RNN is given by an initial state $h_0 in RR^d$ and a map $h_t = f(h_(t-1), y_t)$. An RNN-LM uses $"enc"(y_(<=t)) = h_t$.

#colorbox(title: [Elman RNN])[
  In an Elman RNN we have $f(h_(t-1),y_t) = sigma(U h_(t-1) + V e'(y_t) + b)$, where $U in RR^(d times d), V in RR^(d times R), b in RR^d$ and $e': Sigma mapsto RR^R$ is the embedding function.
]

A softmax RNN is *tight* if for all $t >= N$ (for some $N$) we have $s ||h_t||_2 <= log(t)$, where $s = max_(y in Sigma) ||e(y) - e(EOS)||_2$. In particular, Elman (and Jordan) RNNs with a bounded activation function $sigma$ and the softmax projection function are *tight*.

A *Heaviside Elman RNN* is an Elman RNN using a Heaviside function as non-linearity. Heaviside Elman RNNs (over $overline(RR)$) are equivalent to deterministic PFSAs (this generalizes to any activation function with finite image).

*Minsky's construction* encodes any dPFSA using $U in RR^(|Sigma||Q| times |Sigma||Q|)$ to encode which states are reachable from $h_(t-1)$ and $V in RR^(|Sigma||Q| times |Sigma|)$ to encode which states can be transitioned to using $y_t$ (the hidden state dimensionality can be reduced to $Omega(|Sigma|sqrt(|Q|))$). Satured Sigmoid Elman RNNs are Turing-complete (because they can encode two-stack PDAs). Is thus undecidable whether an RNN-LM is *tight*.

== Transformers

// TODO: transformer definition, attention definition, self-attention, cross-attention, multi-head attention

// TODO: architecture in vasvani et al (where MLP, where layer norm, number of parameters and time complexities)

*Tightness of transformers*: Any transformer using soft attention is *tight* as its layers are continuous and the set of possible inputs to the first layer is compact, making $"enc"$ bounded. If $p_"LN"$ is an *$n$-gram model*, then there exists a transformer $cal(T)$ with $L(p_"LN") = L(cal(T))$.

== Sampling

In *ancestral sampling* we sample $y_t ~ p(dot | y_(<t))$ until $y_t = EOS$. As this may not halt, we can set a max string length. To calibrate $p$ we can postprocess probabilities using a *sampling adapter* function $alpha: Delta^(|Sigma|-1) mapsto Delta^(|Sigma|-1)$. In *top-k sampling* we set $p(y_t | y_(<t)) = 0$ for all but the $K$ most probable tokens (and then renormalize). In *top-p sampling* (or *nucleus sampling*) we only take the top $p%$ of the probability mass (and renormalize).

== Transfer Learning

#colorbox(title: [ELMo], color: silver)[
  // TODO: verify this
  We have a forward and backward LM using $L$ LSTM layers. ELMo representation of token $y_t$ is given by $gamma^("task") sum_(l=0)^L s_l^"task" h_(t l)^"LM"$ where $s_l^"task" >= 0, h_(t l)^"LM" = (arrow(h)_(t l)^"LM", arrow.l(h)_(t l)^"LM")$. $arrow(h)_(t l)^"LM"$ and $arrow.l(h)_(t l)^"LM")$ are the hidden states of the LM layers.
]

#colorbox(title: [BERT], color: silver)[
  BERT is a encoder transformer pretrained using masked language modelling and next sentence prediction.
]

== Parameter Efficient Fine-Tuning

TODO: partial Fine-Tuning

#colorbox(title: [BitFit], color: silver)[
  // TODO: as described in script
]

#colorbox(title: [Adapter tuning], color: silver)[
  // TODO: as described in script
]

#colorbox(title: [LoRA], color: silver)[
  // TODO: as described in script
  Replace weight matrices $W in RR^(d times k)$ with $W arrow.l W + alpha/r B A$, where $B in RR^(d times r), A in RR^(r times k)$.
]

#colorbox(title: [Prefix tuning], color: silver)[
  // TODO: as described in script
]

#colorbox(title: [Diff pruning], color: silver)[
  Learn which parameters to update; learn sparse $delta$ s.t. $theta_"FT" = theta_"LM" + delta$; regularize $delta$ by $L_0$-norm; takes up more GPU memory than ful parameter fine-tuning as new parameters are introduced
]

== RAG

#colorbox(title: [kNN-LM], color: silver)[
// TODO:
Store all embedded prefixes and their following words in a database. At inference time, retrieve the $k$ nearest neighbors of a prefix and normalize the exponentiated distances to a probability distribution $p_xi$ over words. Then sample from a convex combination of $p_xi$ and the original LM. Dynamic Gating: Set the weighting of distributions depending on the prefix.
]

== Alignment

#colorbox(title: [RLHF], color: silver)[
// TODO:
Reinforcement Learning from Human Feedback (RLHF):
1. Collect a dataset of instructions and answers and fine-tune a model on it.
2. Produce comparison data by sampling several model outputs for a given prompt and asking humans to rank them. Train a reward model based on this data.
3. Use PPO to fine-tune the LM (policy) using the reward model as a reward function.
]

== Adverserial Attacks

// TODO

// TODO: model calibration


// TODO: differential privacy definition
