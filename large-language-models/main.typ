#import "template.typ": *
#import "@preview/xarrow:0.3.1": xarrow

#show: project.with(
  authors: (
    (name: "Thomas Gassmann", email: "tgassmann@student.ethz.ch"),
  )
)

#let define = $attach(=, t: "def")$
#let EOS = $#math.script("EOS")$
#show heading: box
#show heading: set text(fill: blue, spacing: 0.2em)

Probability space is a triple $(Omega, cal(F), bb(P))$ where $bb(P)$ is a measure, $bb(P) [Omega] = 1$ and $cal(F) subset.eq cal(P) (Omega)$. To resolve e.g. the paradox of the infinite coin toss, we require $cal(F)$ to be a $sigma$-algebra.
*Infinite coin toss paradox*: $1 = p(Sigma^infinity) = p(union_(omega in Sigma^infinity) {omega}) = sum_(omega in Sigma^infinity) p({omega}) = 0$.

#colorbox(title: [$sigma$ algebra], color: silver)[
  A set $cal(F) subset.eq cal(P) (Omega)$ is called a $sigma$-algebra s.t.: (1) $Omega in cal(F)$, (2) $Sigma in cal(F) arrow.r.double Sigma^complement in cal(F)$, (3) If $Sigma_1, Sigma_2, dots in cal(F)$, then $union.big_(n=1)^infinity Sigma_n in cal(F)$.
]

#colorbox(title: [Probability measure], color: silver)[
  Probability $bb(P)$ over a measure space $(Omega, cal(F))$ is $bb(P): cal(F) arrow [0,1]$ s.t.: (1) $bb(P) (Omega) = 1$, (2) If $Sigma_1, Sigma_2, dots in cal(F)$ is countable seq. of disjoint events, then $bb(P) [union.big_(n=1)^infinity Sigma_n] = sum_(n=1)^infinity bb(P) [Sigma_n]$, i.e. we have a measure space if $Omega$ is a set and $cal(F)$ is a $sigma$-algebra over $Omega$.
]

#colorbox(title: [Measureable function], color: silver)[
  $(Omega, cal(F))$ and $(S, T)$ measure spaces. RV is a measurable function from $Omega arrow S$. A *measurable function* $x: Omega arrow S$ is such that $x^(-1) (Sigma)$ is measurable for $Sigma$ measurable, i.e. $Sigma in T arrow.double x^(-1) (Sigma) in cal(F)$.
]

#colorbox(title: [KL divergence], color: silver)[ For two distributions $P$ and $Q$ we define $D_"KL" (P || Q) = sum_(x in cal(X)) P(x) log(P(x) / (Q(x)))$.
]

#colorbox(title: [Cross entropy], color: silver)[
  For two distributions $P$ and $Q$ over a set $cal(X)$, we define $H(P, Q) = - sum_(x in cal(X)) P(x) log(Q(x))$. The entropy $H(P) = H(P, P)$. Note that $H(P, Q) = H(P) + D_"KL" (P || Q)$.
]

*Precision*: $"TP"/("TP" + "FP") = 1 - "FDR"$; *FDR*: $"FP"/("TP" + "FP")$; *TNR*: $"TN"/("TN" + "FP")$; *TPR/Recall*: $"TP"/("TP" + "FN")$; *FPR/$"error"_1$*: $"FP"/("TN" + "FP")$; *FNR/$"error"_2$*: $"FN"/("TP" + "FN")$; *F1*: $(2 "TP")/(2 "TP" + "FP" + "FN") = 2/(1/"Precision" + 1/"Recall")$; *Precision\@K*: precision of top-K results of a query, i.e. $("TP")/K$; *AP\@K*: average of all the Precision\@K values across all $K$ values, $(sum_(t=1)^K "Precision@t" times "rel"_t)/("#Positives")$, where $"rel"_t$ is an indicator variable if $t$-th element in prediction should actually be retrieved (pos.); *mAP*: mean of the AP\@K metric across all metrics in dataset

== Foundations
An *alphabet* $Sigma$ is a finite, non-empty set. A *string* is a finite sequence of symbols drawn from an alphabet. $Sigma^ast$ is the set of all strings over $Sigma$, and is countable. $Sigma^infinity$ is the set of infinite sequences over $Sigma$. A *language* is a subset of $Sigma^ast$ for some alphabet $Sigma$. A language model is a distribution over $Sigma^ast$.

// #colorbox(title: [Language Model (Informal)], color: silver)[
//   Given an alphabet $Sigma$ and a distinguished $EOS in.not Sigma$, a language model is a collection of conditional probability distributions $p(y | bold(y))$ (probability of $y$ occuring as next token after string $bold(y)$) for $y in Sigma union {EOS}$ and $bold(y) in Sigma^ast$.
// ]

// An *energy function* is a function $hat(p) : Sigma^ast arrow RR$.

#colorbox(title: [Globally normalized model])[
  Let $hat(p)_("GN")(bold(y)) : Sigma^ast arrow RR$ be an energy function. A globally normalized model (GNM) is defined as:
  $p_("LM")(bold(y)) define (exp(- hat(p)_("GN")(bold(y)))) / (sum_(y' in Sigma^ast) exp(-hat(p)_("GN")(bold(y')))) define exp(-hat(p)_"GN" (bold(y)))$ where $Z_"G" = sum_(bold(y') in Sigma^ast) exp(-hat(p)_"GN" (bold(y')))$ is the normalization constant.
]

Any normalizable energy function $hat(p)_"GN"$ ($Z_G$ is finite) induces a language model.

#colorbox(title: [Sequence model], color: silver)[
  For an alphabet $Sigma$ a sequence model is defined as a set of conditional probability distributions $p_"SM"(y | bold(y))$ for $y in overline(Sigma)$ ($overline(Sigma) = Sigma union \{ EOS \}$) $bold(y) in Sigma^ast$. $bold(y)$ is called history/context. That is, we have $\{ p_"SM" (y | bold(y)) \}_(bold(y) in Sigma^ast)$ for $y in overline(Sigma)$.
]

// A sequence model is a probability distribution over $Sigma^ast union Sigma^infinity$.

#colorbox(title: [Locally normalized model /autoregressive model])[
  For $p_"SM"$ a sequence model over $overline(Sigma)$: A locally normalized language model (LNM) over $Sigma$ is defined as: $p_"LN" (bold(y)) define p_"SM" (EOS | bold(y)) product_(t=1)^(|bold(y)|) p_"SM" (y_t |  bold(y)_(<t))$ for $y in Sigma^ast$. LNM is tight if $sum_(bold(y) in Sigma^ast) p_"LN" (bold(y)) = 1$.
]

#colorbox(title: [Prefix probability], color: silver)[
  Let $p_"LM"$ be a language model. The prefix probability of $p_"LM"$ is: $pi (bold(y)) define sum_(y' in Sigma^ast) p_"LM" (bold(y) bold(y'))$, i.e. cumulative probability of all strings in the language beginning with $bold(y)$.
]

Any language model can be locally normalized.
// TODO: should know how to prove this, telescoping product, see page 24

#colorbox(title: [Tightness conditions], color: silver)[
  *(1)* Let $p_EOS (t) = (sum_(omega in Sigma^(t-1)) p_"LN" (omega) p_"LN" (EOS | omega))/(sum_(omega in Sigma^(t-1)) p_"LN" (omega))$. $p_"LN"$ is *tight* iff. $exists t >= 1: p_EOS (t) = 1 or sum_(t=1)^infinity p_EOS (t) = infinity$.
  *(2)* If $p_"LN" (EOS | bold(y)) >= f(t)$ for all $bold(y) in Sigma^t$ (and $t$) and $sum_(t=1)^(infinity) f(t) = infinity$, then $p_"LN"$ is tight.
]

*softmax* is defined as $"softmax"(x)_i = exp(x_i / tau) / (sum_(j=1)^n exp(x_j / tau)))$ for a temperature parameter $tau > 0$. As $t arrow infinity$ the distribution becomes uniform, as $tau arrow 0$ the distribution becomes spiked. We have $"softmax"(x) = "argmax"_(p in Delta^(n-1)) p^top x - tau sum_(i=1)^n p_i log(p_i)$. The *sparsemax* function is defined as $"sparsemax" (x) = "argmin"_(z in Delta^(n-1)) ||z - x||_2^2$. This addresses the drawback of softmax that $"softmax"_i (z) > 0 space forall z, i$ (some tasks prefer sparse probability).

#colorbox(title: [Representation-based Language Model (RBLM)])[
  Embedding matrix $E$ and an encoding function $"enc": Sigma^ast mapsto RR^d$ define a locally normalized language model using the sequence model: $p_"SM" (overline(y)_t | bold(overline(y))_(<t)) = "softmax" (E "enc" (bold(overline(y))_(<t)))_(overline(y)_t)$
]

$s = max_(y in Sigma) ||e(y) - e(EOS)||_2$ and $z(t) = max_(omega in Sigma^t) ||"enc"(omega)||_2$, where $e(dot)$ is the symbol embedding function. *RBLM Tightness:* If $s dot z(t) <= log(t)$ for all $t >= N$ for some $N$, then the induced RBLM is *tight*. In particular, $"enc"(dot)$ is bounded, then model is *tight*.

== Finite State Language Models
#colorbox(title: [FSA])[
  An FSA is a tuple $cal(A) = (Q, Sigma, delta, I, F)$, where $Q$ is a finite set of states, $Sigma$ is a alphabet, $delta subset.eq Q times (Sigma union {epsilon}) times Q$ are the transitions, and $I, F subset.eq Q$ are the initial/final states.
]

#colorbox(title: [Weighted FSA])[
  An WFSA is a tuple $cal(A) = (Q, Sigma, delta, lambda, rho)$, where $delta subset.eq Q times (Sigma union {epsilon}) times RR times Q$ are the (weighted) transitions, and $lambda, rho: Q mapsto RR$ are the initial/final weights.
]

WFSA is *probabilistic* if $lambda, rho$ and the transition weights form are non-negative, $sum_(q in Q) lambda(q) = 1$ and for all $q in Q$ we have $rho(q) + sum_(q xarrow(a "/" w) q') w = 1$. The *weight of a path* $pi = q_1 xarrow(a_1 "/" w_1) q_2 dot dot q_N$ in a WFSA $cal(A)$ is given by $w(pi) = lambda(q_1) product_(i=1)^(N) w_i rho(q_N)$. $Pi(cal(A), y)$ is the set of all paths where $cal(A)$ yields $y$.

The *Allsum* of a WFSA $cal(A)$ is defined as $Z(cal(A)) = sum_(y in Sigma^ast) cal(A) (y) = sum_(y in Sigma^ast) sum_(pi in Pi(A, y)) w(pi) = arrow(lambda) sum_(d=0)^infinity T^d arrow(rho) = arrow(lambda) (I - T)^(-1) arrow(rho)$, where $T$ is the transition matrix of $cal(A)$. *Tightness of PFSA*: A state $q in Q$ is accessible if there exists a non-zero weighted path from an initial state to $q$. It is co-accessible if there exists a non-zero weighted path from $q$ to a final state. A PFSA is *tight* iff. all accessible states are co-accessible.

// CFG skipped

== RNNs
A RNN is given by an initial state $h_0 in RR^d$ and a map $h_t = f(h_(t-1), y_t)$. An RNN-LM uses $"enc"(y_(<=t)) = h_t$.

#colorbox(title: [Elman RNN])[
  In an Elman RNN we have $f(h_(t-1),y_t) = sigma(U h_(t-1) + V e'(y_t) + b)$, where $U in RR^(d times d), V in RR^(d times R), b in RR^d$ and $e': Sigma mapsto RR^R$ is the embedding function.
]

A softmax RNN is *tight* if for all $t >= N$ (for some $N$) we have $s ||h_t||_2 <= log(t)$, where $s = max_(y in Sigma) ||e(y) - e(EOS)||_2$. Elman RNNs with a bounded activation function $sigma$ and the softmax projection function are *tight*. A *Heaviside Elman RNN* is an Elman RNN using a Heaviside function as non-linearity. Heaviside Elman RNNs (over $overline(RR)$) are equivalent to deterministic PFSAs (generalizes to any activation function with finite image). 
*Minsky's construction* encodes dPFSA using $U in RR^(|Sigma||Q| times |Sigma||Q|)$ for which states are reachable from $h_(t-1)$ (_all_ next states) and $V in RR^(|Sigma||Q| times |Sigma|)$ to encode which states can be transitioned to with $y_t$ (all states that can be transitioned to using $y_t$). $bold(b) = -1$ for Heaviside. (hidden state dimensionality can be reduced to $Omega(|Sigma|sqrt(|Q|))$). Satured Sigmoid Elman RNNs are Turing-complete (can encode two-stack PDAs). *Undecidable* whether an RNN-LM is *tight*.

== Transformers
$K,Q,V$ usually no bias.

#colorbox(title: [Attention])[
  Let $f: RR^d times RR^d mapsto RR^d$ be a scoring function (e.g. dot product) and $f_(Delta^(d-1))$ a projection function (e.g. softmax). Let $q in RR^d$, $K_t = (k_1^top, dots, k_t^top) in RR^(t times d)$ and $V_t = (v_1^top, dots, v_t^top) in RR^(t times d)$. Attention over $K_t, V_t$ is a function $"Att"(q, K_t, V_t): RR^d times RR^(t times d) times RR^(t times d) mapsto RR^d$ computing the vector $a$ as follows:
  $
    s_t = (s_1, dots, s_t) = f_(Delta^(d-1)) (f(q, k_1), dots, f(q, k_t))\
    a_t = "Att"(q, K_t, V_t) = s_1 v_1 + dots + s_t v_t
  $
]

$"softmax"((Q K^top)/(sqrt(d)))V$ as the definition of *soft attention*, with the softmax function applied to each row independently and $Q in RR^(n times d), K in RR^(t times d), V in RR^(t times d)$ are functions of the input.

#colorbox(title: [Transformer layer])[
  Let $Q,K,V,O$ be parameterized functions from $RR^d$ to $RR^d$. A transformer $cal(T): RR^(T times d) mapsto RR^(T times d)$ takes as input $X = (x_1^top, dots, x_T^top)$ and returns $Z = (z_1^top, dots, z_T^top) in RR^(T times d)$ s.t. $a_t = "Att"(Q(x_t), K(X_t), V(X_t)) + x_t$ and $z_t = O(a_t) + a_t$ for $t = 1, dots, T$.
]

#colorbox(title: [Multi-head Attention Block])[
  Let $H$ number of heads, $Q_h (dot)$, $K_h (dot)$, $V_h (dot)$ parameterized functions from $RR^(T times d)$ to $RR^(T times d)$ and $f_H: RR^(T H times d) mapsto RR^(T times d)$. A multi-head attention block of input $X$ is defined as:
  
  $f_H ("cat"_(1 <= h <= H) ("sftmx"(Q_h (X) K_h (X)^top)) V_h (X)))$
]

*Position encodings*: Sinuisoidal encodings are $P(k, 2i) = sin(k/(n^(2i\/d)))$ and $P(k, 2i+1) = cos(k/(n^(2i\/d)))$, where $k$ is the position in the sequence and $i$ the dimension. Note that $P(x+k, dot)$ is a linear function of $P(x, dot)$.
*Tightness of transformers*: Any transformer with soft attention is *tight* as layers are continuous and set of possible inputs to the first layer is compact, thus $"enc"$ bounded. If $p_"LN"$ is *$n$-gram model*, there exists a transformer $cal(T)$ with $L(p_"LN") = L(cal(T))$.
*Number of parameters*: embedding/unembedding matrices share weights, embedding: $V D$, layer norm: $2D$, multi-headed attention block with $H$ heads (assuming _no bias_): $D dot D\/H dot 3 dot H + D dot D + 2 dot D$ (with bias $4D^2 + 6D$)

== Sampling
In *ancestral sampling* we sample $y_t ~ p(dot | y_(<t))$ until $y_t = EOS$. May not halt, so set max string length. To calibrate $p$ we can postprocess probabilities using a *sampling adapter* function $alpha: Delta^(|Sigma|-1) mapsto Delta^(|Sigma|-1)$ trading off recall for precision by increasing average sample's quality at expensive of diversity. In *top-k sampling* we set $p(y_t | y_(<t)) = 0$ for all but the $K$ most probable tokens (and then renormalize). In *top-p sampling* (or *nucleus sampling*) we only take the top $p%$ of the probability mass (and renormalize).

== Transfer Learning
In *multi-task learning* we share learned information across multiple tasks, which are learned jointly.
// Process of updating weights of a pretrained model for a new target task is called *fine-tuning*. 

// maybe add RoBERTa, AlBERT, Electra, T5

#colorbox(title: [ELMo], color: silver)[
  Forward and backward LM using $L$ LSTM layers, produces context-dependent representation tokens $y_t$ as $gamma^("task") sum_(l=0)^L s_l^"task" h_(t l)^"LM"$ where $s_l^"task"$ softmax, $h_(t l)^"LM" = (arrow(h)_(t l)^"LM", arrow.l(h)_(t l)^"LM")$. $arrow(h)_(t l)^"LM"$ and $arrow.l(h)_(t l)^"LM")$ are hidden states of LM layers.
]

#grid(
  columns: (13em, auto),
  column-gutter: 1em,
  figure(
    image("encoder-decoder.png", width: 13em, alt: "Encoder-decoder architecture")
  ),
  [
    *CoVE* is similar to ELMo, but only uses the final layer instead of all layers.

    *Others*: T5 and BART are encoder-decoder transformers with bidirectional attention flow for input.
  ]
)

#colorbox(title: [BERT], color: silver)[
  Bidir. Encoder Repr. from Transformers is encoder transformer pretrained using *masked language modelling* and *next sentence prediction*. First token of every sequence is special [CLS] token, final hidden state of this token used as aggregate sentence representation, sentences separated with [SEP] token.
]

*BPE*: repeatedly merge most frequent symbol pair $('A','B')$ with $'A B'$, hyperparam: vocab size; tradeoff with sequence length

== PEFT and Prompting

#colorbox(title: [Diff pruning], color: silver)[
  Learn which parameters to tune (*specification-based method*); learn sparse $delta$ s.t. $theta_"FT" = theta_"LM" + delta$; regularize $delta$ by $L_0$-norm; takes up more GPU memory than ful parameter fine-tuning as new parameters are introduced
]

#colorbox(title: [BitFit], color: silver)[
  Only fine-tune bias terms (attention matrix query bias and middle-of-MLP bias).
]

#colorbox(title: [Adapter tuning], color: silver)[
  Add adapters into model, common to place $h arrow.l h + f(h W_"down") W_"up"$ for non-linearity $f$ after each sublayer (multi-head attention and MLP), needs sequential execution
]

#colorbox(title: [LoRA], color: silver)[
  Replace weight matrices $W in RR^(d times k)$ with $W arrow.l W + alpha/r B A$, where $B in RR^(d times r), A in RR^(r times k)$. Init $A$ with Gaussian, $B$ with zeros. Can be executed in parallel.
]

*Discrete prompts* search for prompts in discrete set of tokens, *continuous prompts* search for prompts in embedding space.

#colorbox(title: [Prefix tuning], color: silver)[
  Continuous, prepend sequence of task-specific vectors to input, optimize $M_phi.alt$ to $"max"_phi.alt sum_(y_i) log P(y_i | h_(<i); theta; phi.alt)$ with $h_(<i) = [h_(<i)^((1)); dots; h_(<i)^((n))]$ copied from $M_phi.alt$ if within prefix and otherwise computed using pre-trained LM.
]

*In-context learning*: emergent behavior, models perform previously unseen tasks in few-shot setting without parameter updates. *Prompting strategies*: chain-of-thought (model generates step-by-step reasoning), least-to-most (problem decomposition, solve separately), program-of-thought (formulate reasoning steps as program); *Self-consistency*: generate variety of output with temp. $T > 0$ and select most frequent ans

== VLMs
Text encoder, vision encoder, fusion module (produce cross-modal representations) and optionally decoder. *Merged attention* concatenates text and image features together (feed into single transformer block), *co-attention* feeds text and image features into separate transformer blocks (then use cross-attention to fuse like in encoder-decoder), *pre-training* via *Masked Language Modeling* (MLM), *Image-Text Matching* (ITM), *Image-Text Contrastive Learning* (ITC, predict $N$ matched pairs from $N^2$ possible image-text pairs, e.g. $cal(L)_"ITC"^(i 2 t) (theta) = -1/N sum_(i=1)^N log(exp(s_(i,i)^(i 2 t) \/ sigma) / (sum_(j=1)^N exp(s_(i,j)^(i 2 t) \/ sigma)))$, where $s_(i,j)^(i 2 t) = v_i^top w_j$ for image and word embeddings, used by e.g. CLIP), *Masked Image Modeling* (MIM).

== RAG
parametric models store knowledge in parameters, non-parametric models externally.

#colorbox(title: [TF-IDF], color: silver)[
  $"tf"_(t,d) = log("count"(t,d)+1)$, $"idf"_t = log(N\/"df"_t)$, $"tf-idf"_(t,d) = "tf"_(t,d) dot "idf"_t$ where $N$ is number of docs, score with norm. cos. sim., after simplification $"score"(q,d) = sum_(t in q) "tf-idf"_(t,d)/(|d|)$, bad: dimension of vectors is same as vocabulary
]

With *dense retrieval*, we use dot product of encoding in embedding space, use contrastive learning to train. *REALM* retrieves texts, concatenates them to input, unlike prototypical RAG jointly optimizes retrieve and predict steps, *RETRO*: fuses artefact into intermediate layer using chunked cross-attention, *kNN-LM*: store embedded prefixes and following words in database, at inference retrieve $k$ nn. of prefix and norm. exp-distances to probability distribution $p_xi$ over words, then sample from convex combination of $p_xi$ and original LM. Dynamic Gating: Set weighting of distributions depending on prefix.

== Alignment
*Log-derivative trick*: $gradient_theta log p(x; theta) = (gradient_theta p(x; theta)) / (p(x; theta))$, can be used to show that $gradient_theta EE_(p(x;theta)) [f(x)] = EE_(p(x;theta)) [gradient_theta log p(x;theta) f(x)]$, which can be approximated using Monte Carlo sampling.


// TODO: ppo

#colorbox(title: [Instruction tuning], color: silver)[
Finetune LM on collection of datasets described via instructions.]

#colorbox(title: [RLHF], color: silver)[
Reinforcement Learning from Human Feedback (RLHF): 
*(1)* Collect a dataset of instructions and answers and fine-tune a model on it.
*(2)* Produce comparison data by sampling several model outputs for a given prompt and asking humans to rank them. Train a reward model based on this data.
*(3)* Use PPO to fine-tune the LM (policy) using the reward model as a reward function.
]

RLHF with PPO is expensive, unstable and sensitive to choice of hyperparams, reward model is also large (expensive to load/compute), *DPO* (Direct Preference Optimization) directly fine-tunes LM to max log-likelihood of preference data, uses *Bardley-Terry* model given by $p(y_w succ y_l) = sigma(r(x,y_w) - r(x, y_l))$, binary classification, minimize negative log-likelihood loss, *best-of-n*: one can also simply overgenerate $n$ samples, rank them and pick the one with highest reward (no LM update)

== Calibration
Ensure models output probability reflects true likelihood of event, let $B_m$ be samples with pred in interval $((m-1)/M), m/M]$, reliability diagram plots $"conf"(B_m) = sum p_i$ vs. $"acc"(B_m) = 1/(|B_m|) sum bold(1)(y_i = hat(y_i))$, we also have $"ECE" = sum (|B_m|)/M | "acc"(B_m) - "conf"(B_m)|$, can regularize calibration during training or rescale using softmax temp.

== Privacy
prevent server from seeing all training data: *secure MPC* or *fully homomorphic encryption*. Still slow and expensive.

#colorbox(title: [Adverserial examples], color: silver)[
  Perturb example with $delta$ to force misclassification, i.e. maximize $L(f_theta (x + delta), y)$ subject to $||delta||_infinity <= epsilon$. This can be solved using *projected gradient descent*.
  Does not work for text as $x + delta$ is unlikely to be a valid token embedding. Solve $"argmax"_v (E_v - x_i)^top gradient_(x_i) L$ and replace $x_i$ with $v$.
]

#colorbox(title: [Federated learning], color: silver)[
  Clients send gradients to central server, but training data can be recovered from gradients. Weight-trap attacks are also possible if the servers sends a model s.t. $gradient_theta L(f_theta (x_i)) = x_i$.
]

#colorbox(title: [Differential privacy])[
  An algorithm $cal(M)$ is $epsilon$-differentially private if for any "neighboring" datasets $D_1, D_2$ differing only in a single element, and any output $S$ we have: *$PP[cal(M)(D_1) in S] <= exp(epsilon) PP[cal(M)(D_2) in S]$*. If $cal(M)$ is $epsilon$-DP, then $f(cal(M))$ for any function $f$ is also $epsilon$-DP. If $cal(M)_1$ is $epsilon_1$-DP and $cal(M)_2$ is $epsilon_2$-DP, then $f(cal(M)_1, cal(M)_2)$ is $(epsilon_1 + epsilon_2)$-DP.
]

// TODO: data memorization
