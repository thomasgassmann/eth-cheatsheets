#import "template.typ": *
#import "@preview/diagraph:0.2.1": *

#show: project.with(
  title: "Visual Computing",
  authors: (
    (name: "Thomas Gassmann", email: "tgassmann@student.ethz.ch"),
  ),
  date: "January 25, 2024",
)

= Computer Vision
== Digital images & sensors
An image (as 2D signal) is a continuous function. A pixel is a discrete sample of that cont. function.

Some challenges with images: transmission interference, compression artifacts, spilling, scratches, sensor noise, low contrast or resultion, motion blur.

#colorbox(title: [Charge coupled device (CCD)])[
  An array of photosites (a bucket of electrical charge) hold charge proportional to incident light intensity during exposure. Charges move down through sensor array, ADC (analog-digital-converter) measures line-by-line.
  - *Blooming*: Oversaturated photosites cause vertical channels to "flood" (bright vertical line)
  - *Bleeding/Smearing*: While charge in transit, bright light hits photosites above - worse with shorter shutter times (electrical shutters only)
  - *Dark current*: CCDs produce thermally generated charge: random noise despite darkness. Avoid by cooling, worsens with age.
  - High *production cost*, high *power consumption*
]

#colorbox(title: [CMOS sensors])[
  Mostly same as CCD, but each sensor has amplifier. Cheaper, lower power, no blooming, on-chip integration with other components, but less sensitive and more noise. Rolling shutter is an issue for sequential read-out of lines. Also susceptible to dark current.
]

We can either sample an image (in 2D) either in a cartesian grid, hexagonally or other non-uniform ways. *Undersampling* means loss of information and introduces aliasing: signals reconstructed incorrectly and higher frequencies incorrenty show up as lower frequency. To reconstruct a continuous image, we could use bilinear interpolation: 

// fixme smaller math and auto-line fix with new typst release
$f(x,y) = (1 - a) (1 - b) dot.c f(i, j) + a (1 - b) dot.c \ f(i + 1, j) + a b dot.c f(i + 1, j + 1) + (1 - a) b \ dot.c f(i + 1, j)$

#colorbox(title: [Nyquist-Shannon sampling theorem], color: silver, inline: false)[
  Sample frequency must be at least twice as big as the highest frequency in the signal / image ($omega_s > 2 dot omega_(max)$, strict inequality)
]

*Quantization*: real-valued function get digital values. #underline[_*Lossy*_] (in contrast to sampling)! Simple version: equally spaced levels with k intervals: $k = 2^b$.

*Geometric resolution*: \#pixels per area \
*Radiometric resolution*: \#bits per pixel

*Image noise*:
Common model for image noise: additive Gaussian. $I(x, y) = f(x, y) + c$ where $c tilde N(0, sigma^2)$ (so density $p(c) = (2 pi sigma^2)^(-1) e^((-c^2) / 2 sigma^2)$).
Poisson (shot) noise: $p(k) = (lambda^k e^(- lambda)) / (k!)$. Rician noise (MRI): $p(I) = 1 / (sigma^2) exp((-(I^2 + f^2)) / (2 sigma^2)) I_0 ((I f) / (sigma^2))$
Multiplicative noise: $I = f + f c$, Salt-and-pepper noise (impulse noise): sparsely occurring white and black pixels, Signal to noise ratio (SNR) s is index of image quality: $s = F / sigma$ where $F = 1 / (X Y) sum_(x = 1)^X sum_(y = 1)^Y f(x, y)$. Peak SNR (PSNR): $s_"peak" = (F_"max") / sigma$ (average signal, divided by stddev of signal).

*Color camera concepts*: Prism (splitting light, 3 sensors, requires good alignment), Filter mosaic (coat filter on sensor, grid, introduces aliasing), Filter wheel (rotate filter in front of lens, static image), CMOS sensor (absorbing colors at different depths), Rolling shutter effect (the effect produced by the sequential readout of pixels while a digital camera is moving)

== Image segmentation
Partition image into regions of interest.

*Complete segmentation* of $I$: regions $R_1, ..., R_N$ s.t. $I = union.big_(i = 1)^N R_i$ and $R_i sect R_j = emptyset "for" i != j$.

#colorbox(title: [Thresholding], color: silver)[
  produces binary image by labeling pixels _in_ or _out_ by comparing to some threshold $T$.
]
$B(x, y) = 1 "if" I(x, y) >= T "else" 0$, finding $T$ with trial and error, compare results with ground truth.

*Chromakeying* choose special background color $bold(x)$, then segmentation using: $I_alpha = norm(bold(I) - bold(x)) > T$. Issues: hard $alpha$-mask, variation not same in 3 channels.

#colorbox(title: [Receiver operating characteristic (ROC)], inline: false)[
  ROC curve describes performance of binary classifier. Plots (y-axis, sensitivity, $"TP" / ("TP" + "FN")$, TPR) against (x-axis, 1-specificity, $"FP" / ("FP" + "TN")$, FPR). We can choose operating point with gradient $beta = N / P dot.c (V_"TN" + C_"FP") / (V_"TP" + C_"FN")$ where $V$ value and $C$ cost.
]

*Pixel neighbourhoods* or 4/8-connectivity: 4-neighb. means horiz. + vert. or 8-neighb. with diag.

#colorbox(title: [Region growing], color: silver)[
  Start from seed point / region, include pixels if some criteria is satisfied (e.g. pixel-pixel threshold or with std-deviation $sigma$, mean $mu$ of graylevels in region: $(I(x, y) - mu)^2 < (n sigma)^2$), iterate until no pixels added.
]
Seed region(s) by hand or conservative thresholding

#colorbox(title: [Snakes], color: silver)[
  Active contour, polygon, where each point on contour moves away from seed while inclusion criteria in neighborhood is met, often smoothness constraint. Iteratively minimize energy: $E = E_"tension" + E_"stiffness" + E_"image"$
]

#colorbox(title: [Distance measures $I_alpha$], inline: false)[
  - Plain BG subtraction: $bold(I)_alpha = abs(bold(I) - bold(I)_"bg")$
  - *Mahalanobis* $bold(I)_alpha = sqrt((bold(I) - mu)^T Sigma^(-1) (bold(I) - mu))$ where $Sigma$ is the BG image's covariance matrix: \
    $Sigma_(i j) = EE[(X_i - mu_i) (X_j - mu_j)^top]$, estimate it from $n >= 3$ data points: $1 / (n - 1) sum_(i = 1)^n (x_i - mu) (x_i - mu)^T$
  Needs thresholds: $bold(I)_alpha > bold(T)$ ($T in RR^3$ RGB, gray $RR$)
]

*Markov random fields*: Cost to assign label to each pixel, cost to assign pair of labels to connected pixels. Solve with graph cuts, source = FG, sink = BG. Minimize energy in polynomial time using MinCut. To get decent results, we need *alpha estimation* / border matting along edges. 
// $E(y; theta, "data") = sum_i (psi_i; theta, "data") + \ sum_(i, j in "edges") psi_2(y_i, y_j; theta, "data")$

== Image filtering
Modify pixels on some function of local neighb.
*Shift-invariant*: same for all pixels, $K$ not dependent on pos.\
*Linear*: linear combination of neighb.
*Local*: Filter is linear, if output only depends on pixels in its neighbors, not on all pixels (global)

#colorbox(title: "Correlation")[ 
  Template matching: $I' = K circle.small I$
  
  $I(x, y) = sum_((i, j) in N(x, y)) K(i, j) I(x + i, y + j)$
]

#colorbox(title: "Convolution")[
  Point spread function: $I' = K convolve I$

  $I'(x, y) = sum_((i, j) in N(x, y)) K(i, j) I(x - i, y - j)$

  Linear, associative, shift-invariant, commutative (if dimensions are identical). For the continuous case: $g(x) = f(x) * k(x) = integral_RR f(a) k(x - a) dif a$
]

The 2 operations are the same with reversed kernels. \
*Filtering near edges*: clip to black, wrap around, copy edge, reflect across edge, vary filter! \
*Separable*: kernel $K(m, n) = f(m) g(n)$, if the kernel matrix has rank 1 it's separable because it's the outer product of two vectors \

#colorbox(title: "Important kernel examples", color: purple, inline: false)[
  #grid(columns: (auto, auto, auto, auto), gutter: 1em,
    [Low-pass \ Mean], $1 / 9 mat(1,1,1; 1,1,1; 1,1,1)$,
    [High-pass], $mat(-1,-1,-1;-1,8,-1;-1,-1,-1)$,
    [Laplacian], $mat(0,1,0; 1,-4,1;0,1,0)$,
    [Prewitt (x)], $mat(-1,0,1;-1,0,1;-1,0,1)$,
    [Gaussian], [\ $G_sigma = 1 / (2 pi sigma^2) e^(-(x^2 + y^2) / (2 sigma^2))$],
    [Sobel (x)], $mat(-1,0,1; -2,0,2; -1,0,1)$,
    [Diff. (x)], $mat(-1, 1)$, [Diff. (y)], $mat(-1, 1)^T$
  )
]
The Gaussian kernel is rot. symmetric, single lobe (neighbor's influence decreases monotonically, also in frequency domain), FT is again a Gaussian, separable, easily made efficient.

Laplacian is rot. invariant (isotropic), usually more noisy since it uses 2nd derivative.

#fitWidth($ (diff f) / (diff x) = lim_(epsilon -> 0) (f(x + epsilon, y) / epsilon - f(x, y) / epsilon) approx (f(x_(n + 1), y) - f(x_n, y)) / (Delta x) $)
Hence, diff. leads to the convolution $mat(-1, 1)$

Image sharpening: enhances edges by increasing high frequency components: $I' = I + alpha abs(k convolve I)$ where $k$ high-pass filter, $alpha in [0, 1]$.

// TODO: integral images?

== Edge detection
#colorbox(title: [Laplacian operator], color: silver)[
  Find 0s in 2nd deriv $I''$ to locate edges. Rot. invariant, yields very noisy but thin and uninterrupted edges. Sensitive, blur first (Laplacian of Gaussian) or suppress edges with low gradient magnitude.
]

$"LoG"(x, y) = - 1/ (pi sigma^4) (1 - (x^2 + y^2) / (2 sigma^2)) exp(-(x^2 + y^2) / (2 sigma^2))$
$nabla^2 f(x, y) = (diff^2 f(x, y)) / (diff x^2) + (diff^2 f(x, y)) / (diff y^2)$, DoG can approximate LoG.

#colorbox(title: [Canny edge detector], color: silver)[
  Thin, uninterrupted edges, no guarantee on the connectivity of edges, extended more completely than with simple thresh.
  + Smooth image with Gaussian filter
  + Compute grad. mag. & orient. (Sobel, Prewitt, ...)
    #fitWidth($ M(x, y) = sqrt(((diff f) / (diff x))^2 + ((diff f) / (diff y))^2), alpha(x, y) = tan^(-1)((diff f) / (diff y) slash.big (diff f) / (diff x)) $)
  + Nonmaxima suppression: quantize edges normal to 4 dirs, if smaller than either neighb. (in given direction) suppress
  + Double thresholding (hysteresis): $T_"high", T_"low"$, keep if $>= T_"high"$ or $>= T_"low"$ and 8-conn. through $>= T_"low"$ to $T_"high"$ px. (first detect strong/weak edge pixels, then reject weak edge pixels not connected with strong edge pixels)
]

#colorbox(title: [Hough transformation], color: silver)[
  Fits a curve (line, circles, ...) to set of edge pixels (e.g. $y = m x + c$)
  + Subdivide $(m, c)$ plane into discrete bins init to 0
  + Draw line in $(m, c)$ plane for each edge pixel $(x, y)$, increment bins by 1 along line
  + Detect peaks in $(m, c)$ plane.
]
Infinite slopes arise, reparameterize line with $(theta, rho)$: $x cos theta + y sin theta = rho$. For circles with known radius: $(x - a)^2 + (y - b)^2 = r^2$, else 3D Hough. _(e.g. for a circle with unknown radius we have a cone in $(x,y,r)$ space, then find maximum in 3D bins)_

*Corner detection*: Edges only well localized in single direction. We need acc. local., invar. against shift, rot., scale, brightness, noise robust, repeatability. We define Local displacement sensitivity: $S(Delta x, Delta y) =$ $sum_((x, y) in "window") (f(x, y) - f(x + Delta x, y + Delta y))^2$. Using the Taylor approx. below and $bold(M)$, we get: $f_x = I_x, ...$

#fitWidth($ f(x + Delta x, y + Delta y) approx f(x, y) + f_x (x, y) Delta x + f_y (x, y) Delta y \
bold(M) = sum_((x, y) in "window") mat(f_x^2 (x, y), f_x (x, y) f_y (x, y); f_x (x, y) f_y (x, y), f_y^2(x, y)) $)

$ S(Delta x, Delta y) approx mat(Delta x, Delta y) bold(M) mat(Delta x, Delta y)^T approx bold(Delta)^T bold(M) bold(Delta) $

#colorbox(title: [Harris corner detection])[
  Compute matrix $bold(M)$. Compute $C(x, y) = det(bold(M)) - k dot.c ("trace"(bold(M)))^2$ $= lambda_1 lambda_2 - k dot.c (lambda_1 + lambda_2)^2$. Mark as corner if $C(x, y) > T$. Do non max-suppression and for better local., weight central pixels with weights  for sum in $bold(M)$: $G(x - x_0, y - y_0, sigma)$. Compute subpixel local. by fitting parabola to cornerness function.
]

#grid(columns: 2, column-gutter: 1cm, [
  #image("harris-corner-detection.png", height: 7em, width: 13em)
], [
    Invariant to shift, rot, brightness offset, not scaling, $(Delta x, Delta y)$ (blue box) constant, ellipses (eigenvectors) rotate but shapes (eigenvalues) remain same
])

Can be made *scale-invariant* by looking for strong responses to *DoG filter* over scale space, then consider loc. max. in both position and scale space (see SIFT).

#colorbox(title: [_Lowe's_ Scale Invariant Feature Tranform (SIFT)],)[
  Used to track feature points over 2 images. Look for strong responses in Difference of Gaussian (DoG) over scale space and position, consider local maxima in both spaces to find blobs. Compute histogram of gradient directions (ignoring gradient mag. bc lighting etc.) at selected scale, pos., rot. by choosing principal direction. Now both pictures are at the same scale & orientation, compare gradient histog. to find matching points. $"DoG"(x, y) = 1 / k e^((x^2 + y^2) / (k sigma)^2) - e^(-(x^2 + y^2) / (sigma^2)), k = sqrt(2)$
]

== Fourier transform
#colorbox(title: [Fourier transform])[
  represents signal in new basis (in amplitudes & phases of constituent sinusoids).

  $F(f(x))(u) = integral_RR f(x) dot.c exp(-i 2 pi u x) dif x$ \
  $F^(-1)(g(u))(x) = integral_RR g(u) dot.c exp(i 2 pi x u) dif u$ \
  $F(f(x, y))(u, v) = integral.double_(RR^2) f(x, y) e^(-i 2 pi (u x + v y)) dif x dif y$
]
Discrete FT: $F = bold(U) f$ where $F$ transformed image, $bold(U)$ FT base, $f$ vectorized image. $F(u, v) = sum_(x = 0)^(M - 1) sum_(y = 0)^(N - 1) f(x, y) dot.c exp(-i 2 pi ((u x)/M + (v y)/N))$, Dual Transform: $f(-x) = F(F(f))(x)$

*Relevant*: $cos(x) = (e^(i x) + e^(-i x)) / 2 space.quad sin(x) = (e^(i x) - e^(-i x)) / (2i)$, $sinc(u) = sin(u) / u$, $integral_(-infinity)^infinity e^(-2 pi i x u) dif x = delta(u)$. \
*Dirac delta*: $delta(x) = 0 "if" x != 0 "else undefined"$. 
Properties: \

- $integral_(-oo)^infinity delta(x) dif x = 1$, $delta(alpha x) = delta(x) / abs(alpha)$ and $delta(-t) = delta(t)$
- $(delta convolve f)(x) = integral_(-infinity)^(infinity) f(t) delta(x - t) d t = f(x)$, e.g. $delta(u - k) convolve f(u) = f(u - k)$

*Sampling*: Mult with seq. of $delta$-fnts, *Fourier Rotation Theorem*: $F[f compose R] = F[f] compose R$ for rotation matrix $R$.

#grid(columns: (auto, auto, auto), column-gutter: 1.5em, row-gutter: 0.8em,
  [*Property*], $bold(f(x)), f(x,y)$, $bold(F(u)), F(u, v)$,
  [Linearity], $alpha f_1(x) + beta f_2(x)$, $alpha F_1(u) + beta F_2(u)$,
  [Duality], $F(x)$, $f(-u)$,
  [Convolut.], $(f * g)(x)$, $F(u) dot.c G(u)$,
  [Product], $f(x) g(x)$, $(F * G)(u)$,
  [Timeshift], $f(x - x_0)$, $e^(-2 pi i u x_0) dot.c F(u)$,
  [Freq. shift], $e^(2 pi i u_0 x) f(x)$, $F(u - u_0)$,
  [Different.], $(dif n) / (dif x^n) f(x)$, $(i / (2 pi) u)^n F(u)$,
  [Multiplic.], $x f(x)$, $i / (2 pi) dif / (dif u) F(u)$,
  [Stretching], $f(a x)$, $frac(1, |a|) F(frac(u, a))$,
  [Deriv. Fourier], $x^n f(x)$, $i^n frac(d^n, "du"^n) F(u)$,
)

#grid(columns: (auto, auto), column-gutter: 1.5em, row-gutter: 0.8em,
  $bold(f(x)), f(x,y)$, $bold(F(u)), F(u, v)$,
  $sin(2 pi u_0 x + 2 pi v_0 y)$, [$1/(2i) (delta(u - u_0, v - v_0) - delta(u + u_0, v + v_0))$],
  $cos(2 pi u_0 x + 2 pi v_0 y)$, [$1/(2) (delta(u - u_0, v - v_0) + delta(u + u_0, v + v_0))$],
  $1$, $delta(x)$,
  [$text("Box")(x) = cases(1 #h(1em) x in [-1/2, 1/2], 0 #h(1em) text("else"))$], [$sinc(u) = sin(pi u) / (pi u) text("(norm. sinc)")$],
  [$h(x,y) = f(x)g(x)$], [$H(u, v) = F(u) G(v)$],
  [$delta(x - x_0)$], [$e^(-2 i pi u x_0)$],
  [$e^(2 i pi(u_0 x + v_0 y))$], [$delta(u - u_0, v - v_0)$],
  [$e^(-a x^2)$], [$sqrt(pi/a) exp(-(pi^2 u^2)/a)$],
)

#image("fourier-transforms.png", height: 20em)

#colorbox(title: [Image restoration])[
Image degradation is applying kernel $h$ to some image $f$. The inverse $tilde(h)$ should compensate: $f(x) -> h(x) -> g(x) -> tilde(h)(x) -> f$. Determine with $F(tilde(h))(u, v) = F(h)(u, v) = 1$. Cancellation of freq., noise amplif. Regularize using $tilde(F)(tilde(h))(u, v) = F(h) slash.big (|F(h)|^2 + epsilon)$         
]

== Unitary transforms (PCA / KL)
Images are vectorized row-by-row. Linear image processing algorithms can be written as $g = F f$. Auto-correl. fun.: $R_"ff" = E[f_i dot.c f_i^H] = (F dot.c F^H) / n$.

*Eigenmatrix*: $Phi$ of autocorrelation matrix $R_"ff"$: $Phi$ is unitary, columns form set of eigenvectors of $R_"ff"$: $R_"ff" Phi = Phi Lambda$ where $Lambda$ is a diag. matrix of eigenvecs. $R_"ff"$ is symmetric nonneg. definite, hence $lambda_i >= 0$, and normal: $R_"ff"^H R_"ff" = R_"ff" R_"ff"^H$.

*Autocorrelation matrix*: $EE[X X^top]$, unlike covariance matrix, we do not subtract the mean here ($EE[(X - mu) (X - mu)^top]$).

#colorbox(title: "Karhunen-Loeve / Principal component anal. ", inline: false)[
  + Normalize to remove brightness var.: $x'_i = x_i / norm(x_i)$
  + Center data by subtracting mean: $x''_i = x'_i - mu, mu = 1 / N sum x'_i$
  + Compute covar. mat.: $Sigma = 1 / (N - 1) sum x''_i dot x''_i^T$
  + Compute eigendecomp. of $Sigma$ by solving $Sigma e = lambda e$ with e.g. SVD ($Sigma = U Lambda U^T$)
  + Define $U_k$ as first k eigenval. of $Sigma$, $U_k = mat(u_1, ..., u_k)$ dirs with largest variance.
  + $"PCA"(x_i) = U_k^T (x_i - mu) = U_k^T dot.c x''_i$
  To decompress, use $"PCA"^(-1)(x_i) =U_k dot.c y_i + mu$ where $y$ is the lower-dimensional representation.
]

#colorbox(title: [PCA storage space], color: green)[
  Given $n$ images of size $x times y$, we want to store the dataset given a budget of $Z$ units of space. What is max number $K$ of princip. comp. allowed? \
  We need to store dataset mean $mu: x times y$, truncated eigenmat. $U_k: (x times y) times K$, compr. imgs. ${y_i}: n times K$
]

=== Eigenfaces

Simple recognition, compare in projected space, find nearest neighbour. Find face by computing reconstr. error and minimizing by varying patch pos. Compress data and visualization. Eigenfaces struggle with lighting differences. Fisherfaces improve this by maximizing between-class scatter, minimzing within-class scatter.

=== LDA / Fisherfaces

Find directions where ratio between/within individual variance is maximized (i.e. minimizing within class difference, maximizing between-class difference).

== JPG & JPEG
+ Convert RGB $->$ YUV (Y luminance / brightness, UV color / chrominance). Humans more sensitive to color, compress colors with chroma subsampling (e.g. color of upper left pixel for 4x4 grid)
+ Split image into 8x8 blocks for each YUV component, apply 2D DCT to it. 64 values, top left = low freq., bottom right = high freq. DCT: variant of DFT, fast implementation, only real values
+ Compress using int. devision with weighted matrix (Quantization table), compress bottom-right. Zig-zag run length encoding followed by Huffman.

High compression (10-100x) and 24-bit colors possible, but artifacts / incorrect colors / aliasing. Edges are softened because sharp edges require high freq. JPEG2000 improves by using Haar transform globally, not just on 8x8 blocks, on successively downsampled image (image pyramid)

*Image pyramids*: iter. applied approx. filter / downsampler. Gaussian pyramid, Laplacian is difference between 2 levels in Gaussian pyramid. \

== Optical flow
Apparent motion of brightness patterns. We set $u = (dif x) / (dif t)$, $v = (dif y) / (dif t)$, $I_x = (diff I) / (diff x)$, $I_y = (diff I) / (diff y)$, $I_t = (diff I) / (diff t)$

Brightness constancy: \
$I(x, y, t) = I(x + (dif x) / (dif t) diff t, y + (dif y) / (dif t) diff t, t + diff t)$

Optical flow constraint: \
$(dif I) / (dif t) = (diff I) / (diff x) (dif x) / (dif t) + (diff I) / (diff y) (dif y) / (dif t) + (diff I) / (diff t) = 0$

*Aperture problem*: when flow is computed for point along linear feature (e.g. edge), not possible to determine exact location of corresponding point in second image, only possible to determine the flow normal to linear feature, 2 unknowns for every pixel $(u, v)$ but only one equation $=> oo$ solutions, opt. flow defines a line in $(u, v)$ space, compute normal flow. Need additional constraints to solve.

#colorbox(title: [Horn & Schunck algorithm])[
  Assumption: values $u(x, y)$, $v(x, y)$ are smooth and change slowly with $x, y$. Minimize $e_s + lambda e_c$ for $lambda > 0$ where

  $e_s = integral.double ((u_x^2 + u_y^2) + (v_x^2 + v_y^2)) dif x dif y$ (smooth.) \
  $e_c = integral.double (I_x u + I_y v + I_t)^2 dif x dif y$ (bright. const.)

  Coupled PDEs solved using iter. methods and finite diffs: $(diff u) / (diff t) = Delta u - lambda (I_x u + I_y v + I_t) I_x$ and  $(diff v) / (diff t) = Delta v- lambda (I_x u + I_y v + I_t) I_y$. Has errors at boundaries / information spreads from corner-type patterns.
]

#colorbox(title: [Lucas-Kanade])[
  Assumption: neighb. in NxM patch $Omega$ have same motion $mat(u, t)^T$ (spatial coherence), small movement, brightness constancy assumption. Minimize energy (using least squares) $E = sum_(x, y in Omega) (I_x (x, y) u + I_y (x, y) v + I_t (x, y))^2$
  
  $mat(sum I_x^2, sum I_x I_y; sum I_x I_y, sum I_y^2) vec(u, v) = -vec(sum I_x I_t, sum I_y I_t)$ _sums over patch $Omega$_\
  Let $M = sum (nabla I) (nabla I)^T$ and $b = mat(-sum I_x I_t, -sum I_y I_t)$
  At each pixel, compute $U$ by solving $M U = b$. $M$ singular / fails if all gradient vec. in same dir (along edge, smooth regions). Works with corners, textures
]

#colorbox(title: [Iterative refinement])[
  Estimate OF with Lucas-Kanade. Warp image using estimated OF. Estimate OF again using warped image. Refine estimate by repeating. Fails if intensity structure  poor / large mov.
]
Gradient method fails when intensity structure within window is poor, displacement large etc.
#colorbox(title: [Coarse-to-fine estimation])[
  Create levels by gradual subsampling. Start at coarsest level, estimate OF, iterate and add until reached finest level. Result is OF at finest level.
]
Still fails if large lighting changes happen.

#colorbox(title: [Affine motion])[each pixel provides 1 lin. constr., 6 global unknowns. Solve LSE. From bright. const. eq.:
  #fitWidth($ I_x (a_1 + a_2 x + a_3 y) + I_y (a_4 + a_5 x + a_6 y) + I_t approx 0 $)
]

*SSD tracking*: Large displacements, extract template around pixel, match within search area, use correlation, choose best match. \
*Bayesian Optical flow*: Some low-level human motion illusions can be explained by adding an uncertainty model to LK. E.g. bright. const. with noise.

== Video compression
Visual perception $<24 "Hz"$. Flicker $>60 "Hz"$ in periphery. *Bloch's law*: if stimulus duration $<= 100 "ms"$, duration and brightness exchangeable. If brightness is halved, double duration. This enforces $>10 "Hz"$ for videos.

Interlaced video format: 2 temporally shifted half images (in bands) increases frequency, reduces spatial resolution. Full image repr. is progressive.

#colorbox(title: [Video compression with temporal redundancy], inline: false)[
  Predict current frame based on previously coded frames. Introducing 3 types of coded frames:
  + I-frame: Intra-coded frame, coded independently
  + P-frame: Predictively-coded based on previously coded (P and I, H.264 can also allow B) frames (e.g. motion vec. + changes)
  + B-frame: Bi-directionally predicted frame, based on both previous and future frames. In older standards B-frames are never used as references for prediction of other frames.
]
Inefficient for many scene changes or high motion. 

#colorbox(title: [Motion-compensated prediction], color: silver)[
  partition video into moving objects -- generally pretty difficult. Practical: *block-matching motion est.*: partition each frame into blocks, describe motion of block / find best matching block in reference frame. No obj. det., good perf. Can be sped up with 3 step log search, and improve precision with half-pixel motions. We encode the residual error (with JPG). Motion vector field (set of motion vec. for each block in frame) is easy to encode, fast, simple, periodic.
]

*MPEG Group of Pictures (GOP)* starts with I-frame, ends with B- ("open") or P-frame ("closed")

== CNN and Radon Transform
Given input $x$, learning target $y$, loss function $cal(L)$ compute kernel weights $theta$ using prediction $f(x, theta)$: $arg min_theta cal(L)(y, f(x, theta))$. Linear classifier $f(x, theta) = W x + b$ where $theta = {W, b}$. Use activation func. $phi.alt$ (sigmoid, *ReLU*, tanh, ...) to introduce non-linearity. Use gradient descent and back propagation (recursive appl. of chain rule to compute gradients) to find $theta$. Transformers, GANs, stable diffusion etc enable modern VC breakthroughs.

Classification: $f(x_1, theta)$ as score, take class with larger score. With $y_i in NN, s_i = f(x_i, theta)$, we get loss function: $cal(L)(y, f(x, theta)) = - sum_(i = 1)^N log (exp(s_(i, y_i)) / (sum_j exp(s_(i, j))))$

Regression: $f(x_1, theta)$ as value, can be used for classification by comparing value. With $y_i in RR^n, s_i = f(x_i, theta)$, we get: $cal(L)(y, f(x, theta)) = sum_(i = 1)^N norm(y_i - s_i)^2$

#colorbox(title: [Radon transform], color: silver)[
  Given object with unknown density $f(x, y)$, find $f$ by sending rays from all dirs through object and measure absorption on the other side. Assume parallel beams for given angle and no spreading of beam. X-Ray along line $L$ at distance $s$ has intensity $I(s)$, travelling $diff s$ reduces intens. by $diff I$. reduction depends on intens. and optical density $u(s)$: $(diff I) / I(s) = -u(s) diff s$. $I_(text("finish")) = I_(text("start")) exp(-R)$ where $R = integral_L u(s) dif s$ for the path through the object $L$. Radon transform of $f(x, y)$: $R f(L) = integral_L f(x) |dif x|$.
  
  With $(x, y) = (rho cos theta - s sin theta, rho sin theta + s cos theta)$ and $rho$ the distance from the object center and $theta$ the angle, $s$ the distance along the line, we get:
  $R(rho, theta) = integral u(x, y) dif s$. We now want to find $u(x, y)$ given $R(rho, theta)$.
]
The continuous case of a radon transform of a function $u$ is:
#fitWidth($ R[u](rho, theta) = integral_(-oo)^infinity integral_(-infinity)^infinity u(x, y) delta(rho - x cos theta - y sin theta) dif x dif y $)

*Properties of RT*: Linear, shifting input shifts the RT (does not affect angle), rotating input rotates RT by same angle, RT of 2D convolution is 1D convolution of RT with respect to $rho$, i.e. $R(f *_"2D" g) = R(f) *_"1D" R(g)$, RHS with fixed $theta$, convolution with respect to $rho$.

*Fourier Slice Theorem*: $G_theta (omega) = F[f](omega cos(theta), omega sin(theta))$ where $G_theta (omega) = integral_(-infinity)^(infinity) R[f](rho, theta)  exp(-2 pi i omega rho) dif rho = F[R[f](rho, theta)](omega)$, 1D FT of projection at fixed angle $theta$ is slice of 2D FT of image.

#image("radon-back-projection.png")

*Back projection algorithm*: Given RT, find $u(x, y)$, apply for all proj. angles $theta$:
+ Measure projection (attenuation) data, get $R[f](rho, theta)$ for fixed $theta$
+ 1D FT of projection data, get $G_(theta) (omega) = F[f](omega cos(theta), omega sin(theta))$ as above
+ 2D inverse FT the above, then sum with previous image (backpropagate)
Requires precise attn. meas., sensitive to noise, unstable, blurring in final image (add high-pass filter in Fourier domain after 2nd step to prevent).

= Computer graphics
== Graphics pipeline

#colorbox[
  + Modeling transform - from object to world-space
  + Viewing transform - from world to camera-space
  + Primitive processing - output primitives from transformed vertcies
  + 3D clipping - remove primitives outside frustum
  + Projection to screen-space
  + Scan conversion - Discretize continuous primitives, interpolate attributes at covered samples
  + Lighting, shading, texturing - compute colors
  + Occlusion handling - update color (e.g. z-buffers)
  + Display
]

Contemporary pipeline: CPU, Vector processing (per-vertex ops, transforms, lighting flow control), Rasterization, Fragment processing (per-fragment ops, shading, texturing, blending), Display. Or: \

#render("
  digraph {
    graph [K=0.8]
    node [shape=record]
    
    att -> vs 
    vs -> vpv
    u -> vs
    vpv -> ip
    ip -> vpf
    vpf -> fs
    u -> fs
    fs -> fc

    // Following edges are all invisible - for layout
    Edge [ style = invis ]
    att -> vpv 
    vpv -> vpf
    vpf -> fc
    vs -> ip
    ip -> fs
    ip -> u

  }", labels: (:
    att: "Attributes\n(per-vertex)",
    vs: "Vertex shader",
    vpv: "Varying\n(per-vertex)",
    ip: "Interpolation",
    vpf: "Verying\n(per-fragment)",
    u: "Uniform\nconstants",
    fs: "Fragment shader",
    fc: "Fragment color\n(per-fragment)"
  ),
  width: 90%,
  engine: "sfdp"
)


== Light & Colors

#grid(columns: 2, column-gutter: 0.5em, [
  #image("ciergb.png", height: 15em)
], [
  Light is mixture of many wavelengths. Consider $P(lambda)$ as intensity at wavelength $lambda$. Humans project inf. dimens. to 3D color (RGB). CIE experiment: some colors are not comb. of RGB. (neg. red needed)
])

#colorbox(title: [Color spaces], inline: false)[
  - *RGB* (useful for displays, RGB colors specified)
  - *CIE XYZ*: Change of basis to avoid neg. comp. From xyZ via: $X = (x Y)/y, Y = Y, Z = Y/y - (x Y) /y - Y$.
  - *CIE xyY*: Chomaticity (color) $(x, y)$ derived by normalizing XYZ color components: $x = X / (X + Y + Z), y = Y / (X + Y + Z)$. Y is brightness, $z = 1 - x - y$.
  - *CIE RGB*: (435.8, 546.1, 700.0nm). Linear combination span triangle, the Color Gamut.
  - *CMY*: inverse (subtr.) to RGB. CMY = 1 - RGB.
  - *YIQ*: Luminance Y, In-phase I (orange-blue), Quadrature Q (purple-green). $ vec(Y, I, Q) = mat(0.299, 0.587, 0.114; 0.596, -0.275, -0.321; 0.212, -0.523, 0.311) vec(R, G, B) $ NTSC-Norm used in television, based on psycho physical properties of eye, color space resolution is used for tones which can best be distinguished by eye.
  - *HSV*: hue (base color), saturation (purity of color), value / lightness / brightness (intuitive), easy to pick color, used in e.g. arts
  - *HLS/HSL*: alternative color space to HSV, used for same applications
  - *CIELAB / CIELUV*: color space is perceptually uniform, correct the CIE chart colors to adjust for perceived "distance" betw. colors (small change in euclidean distance $arrow$ small change in perceived color), nonlinear warp. MacAdams ellipses nearly circular.
]

#colorbox(color: yellow, title: [White point calibration])[
  #fitWidth($ vec(x, y, z) = mat(x_R C_R, x_G C_G, x_B C_B; y_R C_R, y_G C_G, y_B C_B; (1 - x_R - y_R) C_R, (1 - x_G - y_G) C_G, (1 - x_B - y_B) C_B) vec(R, G, B)$)
  Set $(R, G, B) = (1, 1, 1)$. Map to given white point in xyz (e.g. $(0.9505, 1, 1.0890)$)), then find $C_R, C_G, C_B$.
]


== Transformations
Change position & orientation of objects, project to screen, animating objects, ... *Homogeneous coordinates* can represent affine maps (translation) with mat.-mul. Add dimension, project vertices $mat(x, y, z, w)^T$ onto $mat(x / w, y / w, z / w, 1)^T$.

#colorbox(color: silver)[
  #grid(columns: (auto, auto, auto, auto), column-gutter: (0em, 1em, 0em), row-gutter: 0.8em,
    [Trans.], $mat(1, 0, 0, t_x; 0, 1, 0, t_y; 0, 0, 1, t_z; 0, 0, 0, 1)$,
    [Scale], $mat(s_x, 0, 0, 0; 0, s_y, 0, 0; 0, 0, s_z, 0; 0, 0, 0, 1)$,
    [Rot. \ x], $mat(1, 0, 0, 0; 0, cos theta, -sin theta, 0; 0, sin theta, cos theta, 0; 0, 0, 0, 1)$,
    [Rot. \ y], $mat(cos theta, 0, sin theta, 0; 0, 0, 1, 0; -sin theta, 0, cos theta, 0; 0, 0, 0, 1)$,
    [Rot. \ z], $mat(cos theta, -sin theta, 0, 0; sin theta, cos theta, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1)$,
    [Shear \ x], $mat(1, 0, "sh"_x, 0; 0, 1, "sh"_y, 0; 0, 0, 1, 0; 0, 0, 0, 1)$,
    [Rot \ (2D)], $mat(cos theta, -sin theta, 0; sin theta, cos theta, 0; 0, 0, 1)$,
    [Shear \ (2D)], $mat(1, a, 0; 0, 1, 0; 0, 0, 1)$
  )
]
*Rigid transforms*: translation, rotation, (sometimes includes reflection). *Linear*: Rotation, Scaling, Shear.  *Projective transforms*: Rigid + Linear + Persp. + Paral.

*Commutativity* ($M_1 M_2 = M_2 M_1$) holds for: ($M_1$ translation, $M_2$ translation), ($M_1$ rotation, $M_2$ rotation, only in 2D), ($M_1$ scaling, $M_2$ scaling), ($M_1$ scaling, $M_2$ rotation)

#colorbox(title: [Change of coord. system], color: purple, inline: false)[
  #grid(columns: (10em, auto),
    image("change-coords.png"),
    $ p' = underbrace(mat(bold(r_1), bold(r_2), bold(r_3), bold(t); 0, 0, 0, 1), bold(M)) mat(p_x; p_y; p_z; 1) $
  )
  
  Transforming a normal: $bold(n') = (bold(M)^(-1))^top bold(n) = (M^top)^(-1)n$
]

#grid(columns: 2, column-gutter: 1em, [
  Parallel (orthographic) projection:
  $
    bold(M_"ort") = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 1)
  $
], [
  Perspective projection:

  $
  bold(M_"per") = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 1 / d, 0)
  $
])


#image("perspective-projection.png", width: 20em)

#colorbox(title: [Quaternions], color: orange)[
  Similar to $CC$, define $i^2 = j^2 = k^2 = - 1$, $i j k = -1$, $i j = k$, $j i = -k$, $j k = i$, $k j = -i$, $k i = j$, $i k = -j$. For $q = a + b i + c j + d k$, we have: \ 
  - $norm(q) = sqrt(a^2 + b^2 + c^2 + d^2)$, $z overline(z) = norm(z)^2$
  - $overline(z) = a - b i - c j - d k$ and $z^(-1) = overline(z) slash.big norm(z)^2$
  - For $z_1 = s_1 + v_1, z_2 = s_2 + v_2$ (where $s_i in RR$) we have $z_1 z_2 = s_1 s_2 - v_1 dot v_2 + s_1 v_2 + s_2 v_1 + v_1 times v_2$ where last term is sum of all components of cross product vector
]
Rotating a point $p = mat(x, y, z)^T$ around axis $u = mat(u_1, u_2, u_3)$ by angle $theta$.
+ Convert $p$ to quaternion $p_Q = x i + y j + z k$
+ Convert $u$ to quaternion $q'' = u_1 i + u_2 j + u_3 k$, normalize $q' = q'' slash.big norm(q'')$
+ Rotate quaternion $q = cos(theta / 2) + q' sin(theta / 2)$ and $q^(-1) = cos(theta / 2) - q' sin(theta / 2)$
+ Rotated point $p' = q p q^(-1)$. Convert to cartesian.

== Lighting & Shading
Lighting: Modelling physical interactions between materials and light sources \
Shading: Process of determining color of pixel

*Solid angle*: $Omega = A / r^2$ steradians (where $r$ radius) \
*Zenith, Azimuth*: Point $omega = (theta, phi.alt)$ on unit sphere:
$omega_x = sin(theta) cos(phi.alt), omega_y = sin(theta) sin(phi.alt), omega_z = cos(theta)$ \
*Energy of photon*: $(h c) / lambda "J" ("kg" dot.c "m"^2 / s^2)$

#colorbox(title: [Basic quanitities of radiometry], inline: false)[
  *Flux*: $Phi(A)$ total amount of energy passing through surface or space per unit time $["J " / "s " = "W "]$ \
  *Luminous flux*: $integral P(lambda) V(lambda) dif lambda$. Perceived power of light weighted by human sensitiv. $["lumen"]$ \
  *Irradiance*: $E(x) = (dif Phi(A)) / (dif A(x))$ flux per unit area arriving at surface $["W " / "m "^2]$. \
  *Radiosity*: $B(x) = (dif Phi(A)) / (dif A(x))$ flux per unit area leaving a surface $["W " / "m "^2]$ \
  *Radient / Luminous Intensity*: $I(arrow(omega)) = (dif Phi) / (dif arrow(omega))$ outgoing flux per solid angle $["W " / "sr"]$. $Phi = integral_(S^2) I(arrow(omega)) dif arrow(omega)$\
  *Radiance*: $L(x, arrow(omega)) = (dif I(arrow(omega))) / (dif A(x)) = (dif^2 Phi(A)) / (dif arrow(omega) dif A(x) cos theta)$ flux per solid angle per perp. area = intensity per unit area
]

*Lamberts cosine law*: Irradiance at surface is proportional to cosine of angle of incident light and surface normal: $E = (Phi slash.big A) cos theta$\
*Bidir. reflectance distr. func.* (BRDF): relation between incident radiance and diff. refl. radiance. \
$f_r (x, arrow(omega)_i, arrow(omega)_r) = (dif L_r (x, arrow(omega)_r)) / (dif E_i (x arrow(omega)_i)) = (dif L_r (x, arrow(omega)_r)) / (L_i (x, arrow(omega)_i) cos theta_i dif arrow(omega)_i)$ \
*Reflection equation*: reflected radiance due to incident illumination from all directions (from BRDF): $integral_(H^2) f_r (x, arrow(omega)_i, arrow(omega)_r) L_i (x, arrow(omega)_i) cos theta_i d arrow(omega)_i = L_r (x, arrow(omega)_r)$ \
*Types of reflections*: Ideal specular (perfect mirror), ideal diffuse (uniform reflection all directions), Glossy specular (majority light distributed in reflection direction), retro-reflective (reflects light back towards source)\
*Attenuation*: $f_"att" = 1 / d_L^2$ due to spatial radiation, loss of flux when light travels through a medium.  *Types*: local illumination only considers the light hitting an object directly from the lightsource, global illumination also considers indirect light bouncing off from other objects that are hitting the object.

#colorbox(title: [Phong Illumination Model], inline: false)[
  Approximate specular reflection by cosine powers
  #fitWidth(
    $ I_lambda = underbrace(I_a_lambda k_a O_d_lambda, "Ambient") + f_"att" I_(p_lambda) [underbrace(k_d O_(d_lambda)(N dot.c L), "Diffuse") + underbrace(k_s (R dot.c V)^n, "Specular")] $
  )

  $I_a$ ambient light intensity, $k_a$ ambient light coefficient, $I_p$ directed light source intensity, $k_d$ diffuse reflection coefficient, $theta in [0, pi / 2]$ angle surface normal $N$ and light source vector $L$, attenuation factor $f_"att"$, $O_(d lambda)$ value of spectrum of object color at the point $lambda$, $R$ is $L$ reflected along the normal $N$ ($R = 2N(N dot L) - L$, i.e. perfect reflection), $V$ is the direction pointing towards the viewer, $n$ determines shape of highlight.

  $k_a, k_d, k_s, n$ are material dependent constants. Increasing $n$ causes the highlight to appear smaller in terms of area (sharp, focused, highlights). As the power increase, more values are
mapped to zero.
]

== Shading models

Flat shading: one color per primitive/polygon

#colorbox(title: [Gouraud Shading], color: gray)[
  Lin.interpol. vertex intensities
  
  + Calculate face normals
  + Calculate vertex normals by averaging _(weighted by angle)_
  + Evaluate illumination model for each vertex
  + Interpolate vertex colors bilinearly on the current scan line
  Problems with scan line interpolation are perspective distortion, orientation dependence, shared vertices, incorrect highlights. Quality depends on size of polygons.
]

#colorbox(title: [Phong Shading], color: gray)[
  Lin. interpol. of vertex normals, barycentric interpolation of normals on triangles.
  #image("phong-shading.png")
  Properties:
    Lagrange: $x = a => n_x = n_a$ \
    Partition of unity: $lambda_a + lambda_b + lambda_c = 1$ \
    Reproduction: $lambda_a dot.c a + lambda_b dot.c b + lambda_c dot.c c = x$.

  Problems: normal not defined / representative.
]

Flat, Gouraud in screen space, Phong in obj. space, e.g. to get Gouraud shading we can move the code from Phong shading from fragment shader to vertex shader (only calculate for each vertex, then interpolate).

*Transparency*: (2 obj., $P_1$ & $P_2$). Perceived intensity $I_lambda = I'_lambda_1 + I'_lambda_2$ where $I'_lambda_1$ is emission of $P_1$ and $I'_lambda_2$ is intensity filtered by $P_1$. We model it as follows: $I_lambda = I_(lambda_1) alpha_1 Delta t + I_(lambda_2) e^(-alpha_1 Delta t)$ where $alpha$ absorption, $Delta t$ thickness. \

#grid(columns: 2, [
  #image("transparency.png", height: 7em, width: 13em)
], [
  Linearization: $I_lambda = I_lambda_1 alpha_1 Delta t + I_lambda_2 (1 - alpha_1 Delta t)$. If last object, set $Delta t = 1$. Problem: rendering order, sorted traversal of polygons and back-to-front rendering.
])

*Back-to-front* order not always clear, resort to depth peeling, multiple passes, each pass renders next closest fragment.

We need more advanced lighting models for metal objects (Cook-Torrence), replacing specular term, with reflection from micro facets, self-shadowing...


== Geometry and textures
Considerations: Storage, acquisition of shapes, creation of shapes, editing shapes, rendering shapes.

*Explicit* repr. can easily model complex shapes, and sample points, but take lots of storage.
- *Subdivision surface*, define surfaces by primitives, use recursive algorithm for refining
- *Point set surfaces*, store points as surface
- *Polygonal meshes*, explicit set of vertices with position, possibly with additional information. Intersection of polygons: either empty, vertex, edge.
- triangle meshes, NURBS, etc.
*Implicit* repr. easily test inside / outside, compact storage, but sampling expensive, hard to model
- algebraic surfaces, constructive solid geometry, level set methods, blobby surfaces, fractals
Need to store textures as bitmaps, hence param. complex surfaces.

// TODO: Laplacian pyramids: information at scale and orientation

*Manifolds*: surface homeomorphic to disk, closed manifolds divids space into two., in manifold mesh there are at most two faces sharing an edge and each vertex has a 1-connected ring of faces around it (or 1-connected half-ring (i.e. boundary)).

*Mesh data structures*: Locations, how vertices are connected, attributes such as normals, color etc. Must support rendering, geometry queries, modifications. E.g. *Triangle list* (list of 3 points, redundant, e.g. STL) or *Indexed Face set* (array of vertices + list of indices, e.g. OBJ, OFF, WRL, costly queries, modifications).

#colorbox(title: [Texture mapping])[
  Enhance details without additional geom. complexity. Map Texture $(u, v)$ coords to geom. $(x, y, z)$ coords. Issues: aliasing, level-of-detail, (e.g. sphere mapping: $(u, v) -> (sin u sin v, cos v, cos u sin v)$). We want low-distortion, bijective mapping, efficiency.
]
*Bandlimiting*: Restrict amplitude of spectrum to zero for frequency beyond cut-off frequency. \

Anti-aliasing filters: 
- Gaussian (low-pass, separable). Closer pixels have higher weight
- Sinc $S_omega_c (x, y) = (omega_c sinc((omega_c x) / pi)) / pi times (omega_c sinc((omega_c y) / pi)) / pi$ ideal low-pass filter, IIR filter. $omega_c$ cutoff freq. Hard to implement, has infinite impulse respons.
- B-Spline: Allow for locally adaptive smoothing, adapt size of kernel to signal chars.

Magnification: for pixels mapping to area larger than pixel (jaggies), use bilinear interpolation.

*Mipmapping*: Store texture at multiple resolutions, choose based on projected size of triangle (far away $->$ lower res, interpol. possible), texels in larger level of mipmap hierarchy represent larger regions of texture (e.g. we halve resolution at each level). Avoids aliasing, improves efficiency, higher storage overhead. (minification)

*Supersampling*: Multiple color samples per pixel, final color is average - different patterns of samples possible (uniform, jittering, stochastic, Poisson).

#colorbox(color: silver, inline: false)[
  *Light map*: Simulate effect of local light source. Can be pre-computed and dynamically adapted \
  *Environment map*: Mirror environment with imaginary sphere or cube for reflective objects \
  *Bump map*: Perturb surface normal according to texture, represents small-scale geometry. Limitations: no bumps on silhouette, no self-occlusions, no self-shadowing. Height stored as grayscale textures. \
  *Displacement mapping*: compared to bump map, displaces geometry, uses height map to displace points along surface normal
]

*Procedural texture*: Generate texture from noise (Perlin, Gabor) from Gaussian pyramid of noise and summing layers with weights.

*Perspective interpolation* in world-space can yield non-linear variation in screen-space. Optimal resampling filter is hence spatially variant.

#image("image.png")

== Scan conversion (rasterization)
Generate discrete pixel values - approxiate with finit amount. \
*Bresenham* line: choose closest pixel at each intersection. Fast decision, criterion based on midpoint $m$ and intersection point $q$. After computing first value $d$, we only need addition, bitshifts. $d_"new" = d_"old" + a$
#image("bresenham-line.png")

*Scan conversion of polygons*:
+ Calculate intersections on scan line
+ Sort intersection points by ascending x-coords.
+ Fill spans between two consecutive intersection points if parity in interval is odd (so inside)

== Bézier Curves & Splines
#colorbox(title: [Bézier Curves])[
  $bold(x)(t) = bold(b)_0 B_0^n (t) + ... + bold(b)_n B_n^n (t)$ where $B_i^n$ are the Bernstein polyn. ($binom(n, i) = (n!) / (i! (n- i)!)$): $B_i^n (t) = binom(n, i) t^i (1 - t)^(n - i) "if" i != 0 "else" 0$ (partition of unity, pos. definite, recursion, symmetry). Coefficients $bold(b)_0, ..., bold(b)_n$ are called control- / Bézier-points, together define the control polygon.

  *Degree* is highest order of polyn., order is deg. + 1. Bézier-Curves have *affine invariance* (affine transf. of curve accompl. by affine transf. of control pts.). Curve lies in *convex hull* of control polygon $"conv"(P) := {sum_(i = 1)^n lambda_i bold(p)_i | lambda_i >= 0, sum_(i = 1)^n lambda_i = 1}$. Control polyg. gives *rough sketch* of curve. Since $B_0^n (0) = B_n^n (1) = 1$, *interpol. endpoints* $bold(b)_0, bold(b)_n$. *Max. number of intersect*. of line with curve is $<=$ number of intersect. with control polygon.
]

Disadvantages: global support of basis functions, new control pts yields higher degree, $C^r$ continuity between segments of Bézier-Curves.

*deCasteljau*: Compute a triangular representation, successively interpolate, "corner cutting":
  #grid(columns: 2, column-gutter: 0.5cm, [
    #table(
      columns: 4,
      rows: 4,
      stroke: none,
      [$b_0$], [], [], [],
      [$b_1$], [$b_0^1$], [], [],
      [$b_2$], [$b_1^1$], [$b_0^2$], [],
      [$b_3$], [$b_2^1$], [$b_1^2$], [$b_0^3$]
    )
  ], [
    Consider the three control points $b_0, b_1, b_2$:
    
    $
    b_0^1(t) = (1 - t)b_0 + t b_1\
    b_1^1(t) = (1 - t)b_1 + t b_2\
    b_0^2(t) = (1 - t)b_0^1(t) + t b_1^1(t)
    $
  ])

#colorbox(title: [B-Spline functions])[
  B-Spline curve $bold(s)(u)$ built from piecewise polyn. bases $s(u) = sum_(i = 0)^k bold(d)_i N_i^n (u)$ \
  Coefficients $bold(d)_i$ are called "de Boor" pts. Bases are piecewise, recursively def. polyn. over sequence of knots $u_0 < u_1 < u_2 < ...$ defined by knot vec. $T = mat(u_0, ..., u_(k + n + 1))$

  $
  N_i^n (u) = N_i^(n-1)(u) (u - u_i) / (u_(i + n) - u_i) + N_(i + 1)^(n - 1)(u) (u_(i + n + 1) - u) / (u_(i + n + 1) - u_(i + 1))
  $

  $
    N_i^0 (u) = cases(1 text("  ") u in [u_i, u_(i + 1)], 0 text("   else"))
  $

  Partition of unity $sum_i^k N_i^n (u) = 1$, positivity $N_i^n (u) >= 0$, compact support $N_i^n (u) = 0$, $forall u in.not [u_i, u_(i + n + 1)]$, continuity $N_i^n$ is $(n - 1)$ times cont. differentiable, local control, affine invariant.
]

*Example:* $N_i^1 (u) = cases((u - u_i) / (u_(i + 1) - u_i) #h(0.5cm) u in [u_i, u_(i+1)], (u_(i + 2) - u) / (u_(i + 2) - u_(i + 1)) #h(0.25cm) u in [u_(i+1), u_(i+2)])$

*deBoor algorithm*: generalizes deCasteljau, evaluate B-spline of degree $n$ at $u$, set $d_i^0 = d_i$, finally $d_0^n = s(u)$
$
d_i^k = (1 - a_i^k) d_(i)^(k-1) + a_i^k d_(i+1)^(k-1), space space space a_i^k = (u-u_(i+k-1))/(u_(i+n) - u_(i+k-1))
$
Note that $a_i^k$ vanishes outside of $[u_(i+k-1), u_(i+n)]$!

== Subdivision surfaces

Generalization of spline curves / surfaces allowing arbitrary control meshes using successive refinement (subdivision), converging to smooth limit surfaces, connecting splines and meshes. 

- *interpolating* (new points are interpolation of control points) vs. *approximating* (control points moved, new points added in-between)
- *primal* (faces are split) vs. *dual* (vertices are split)

=== Loop subdivision

// TODO: Corner-Cutting, Doo-Sabin, Catmull-Clark Subdivision (quadrilateral meshes), Loop Subdivision (triangular meshes)

== Visibility and shadows
*Painter's algorithm*: Render objects from furthest to nearest. Issues with cyclic overlaps &intersec. \
*Z-Buffer*: store depth to nearest obj for each px. Initialize to $oo$, then iter. over polygons and update z-buffer. Limited resolution, non-linear, place far from camera \
*Planar shadows*: draw projection of obj. on ground. No self-shadows, no shadows on other objects, curved surfaces. \
*Projective texture shadows*: Separate obstacles and receivers, compute b/w img. of of obst. from light, use as projective textrue. Need to specify obstacle & receiver, no self-shadows. \
#colorbox(title: [Shadow maps])[
  Compute depths from light & camera. For each px on camera plane, compute point in world coords., project onto light plane, compare $d(bold(x)_L)$ (dist. from light source to projected point on light plane) to $z_L$ (dist light source to $x_L$). If $d(bold(x)_L) < z_L$, then $x$ in shadow. Bias needed to avoid self-shadowing ($d(bold(x)_L) + "b" < z_L$). Point to shadow can be outside field of view of (use cubical shadow map or spot lights). Aliasing due to undersampling shadow map (filter result of depth test).
]
*Shadow volumes*: Explicitly represent volume of space in shadow. If polygon in volume, it is in shadow. Shoot ray from camera, increment each time boundary of shadow volume is intersected. If counter $> 0$, primitive is in shadow. Optimization: use silhouette edges only. Introduces a lot of new geometry, expensive to rasterize long skinny triangles. Objects must be watertight for silhouette optimizations. Rasterization of polygons must not overlap & not have a gap.


== Ray Tracing
#colorbox()[
  Send rays into scene to determine color of px. On obj. hit, send mult. rays (diffused, reflected, refracted) further until we hit light source, or reach some amount of bounces. Figure out whether point in shadow by shooting rays to all light sources. For anti-aliasing, use mult. rays per px. (Pipel.: Ray Generation, Intersection, Shading)
]
*Ray-surface intersections*: Ray equation $bold(r)(t) = bold(o) + t bold(d)$. *Sphere*: $norm(bold(x) - bold(c)) - r^2 = 0$ where $bold(x)$ point of interest, $bold(c)$ center, $r$ radius. Solve for $t$: $norm(bold(o) + t bold(d) - bold(c))^2 - r^2 = 0$. *Triangle*: Barycentric coords. $bold(x) = s_1 bold(p)_1 + s_2 bold(p)_2 + s_3 bold(p)_3$. Intersect: $(bold(o) + t bold(d) - bold(p)_1) dot.c n = 0$. Using the following: $bold(n) = (bold(p)_2 - bold(p)_1) times (bold(p)_3 - bold(p)_1)$ we get $t = - ((bold(o) - bold(p)_1) dot.c bold(n)) / (bold(d) dot.c bold(n))$. Now compute the coeffs. $bold(s)_i$. Test whether $s_1 + s_2 + s_3 = 1$ and $0 <= s_i <= 1$. If so, inside triangle.

*Ray-tracing shading extensions*: Refraction, mult. lights, area lights for soft shadows, motion blur (sample objs, intersect in time), depth of field, cost: $O(N_x N_y N_o) = "#px" dot.c "#objects"$

*Accelerate*: Accelerate with less intersections, introduce uniform grids or space partitioning trees. \
*Uniform grids*: Preprocess: compute bounding box, set grid res., rasterize objects, store refs. to objects. Traversal: incrementally rasterize rays - stop at intersection. Fast & easy, but non-adaptive to scene geometry.
*Space partitioning trees*: octree, kd-tree, bsp-tree, another solution: bounding volume hierarchies (BVH)

// TODO: space partitioning trees, K-D tree, octree

*Model matrix*: from object space to world coordinates, *View matrix*: from world coordinates to camera coordinates, *Projection matrix*: from camera coordinates to screen space

+ All 3D rot. matrices $R$ have property $R^(-1) = R^T$
+ Bézier curves are special cases of B-spline curves
+ Color buffer is updated only when depth test passed
+ Radiance is constant along a ray (in a vacuum)
+ Pinhole camera measures radiance
+ $f_r (omega_i, omega_o) = f_r (omega_o, omega_i)$ is true for any valid BRDF function
+ Given material with a BRDF fct that satisfies $integral_Omega f_r (omega_i, omega_o) dif omega_i = 1$, all incom. energy is reflected
+ For a perfect mirror material, its $f_r (omega_i, omega_o)$ is non-zero $<=> omega_i$ is reflection vec. of $omega_o$ against surf. normal at the point of interest
+ Due to persp. proj., barycentric coords. of values on a triangle of different depths are not an affine function of screen space positions.
