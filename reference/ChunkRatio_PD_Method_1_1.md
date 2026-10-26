# ChunkRatio-PD: Preference Distillation with Chunk Ratios and Mean Residual Correction

**Consolidated method specification, version 1.1 — 26 October 2026**  
**Language:** English.  
**Status:** formulation and evaluation plan fixed for implementation; empirical effectiveness remains unverified.

## 1. Decision and scope

This document consolidates the preceding ChunkRatio-PD development into one method. The agreed core is:

> Transfer teacher alignment changes through comparisons of chunks sharing a prefix, and approximate the reference-transfer correction with one scalar predictor of an expected future log-ratio.

The sequence objective, teacher-pair signal, sibling comparisons, and soft preference loss are retained. Version 1.1 corrects the direction of the mean approximation: use the expected **teacher alignment log-ratio under the frozen student reference**, rather than the expected reference-mismatch log-ratio under the aligned teacher. This targets the regime of small alignment displacement even when the student and teacher bases differ substantially. It does not assume that this regime holds for every model pair. The exact residual remains the theoretical target and a diagnostic benchmark. The method name remains **ChunkRatio-PD**.

The implementation and main experiments should follow this specification. Future questions should clarify or evaluate it. A change to the core method requires a demonstrated mathematical error, a material implementation constraint, or experimental evidence against a stated claim. A new paper, an alternative loss, or a possible extra theorem is not by itself a reason to redesign the method.

### Fixed design

| Component | Decision |
|---|---|
| Teacher access | An aligned teacher $T^+$ and its actual reference $T^0$, with token likelihood access |
| Student reference | Frozen student initialization $\rho$ |
| Initial setting | Shared tokenizer, matched stopping convention, finite response horizon |
| Transfer scale | Unscaled teacher log-ratio, corresponding to the previously discussed $\gamma=1$ |
| Training unit | Two sibling chunks from the same prefix |
| Chunk rule | Fixed token length with EOS/horizon truncation; other causal rules are ablations |
| Teacher target | Analytic teacher chunk log-ratio plus expected future teacher displacement under $\rho$ |
| Correction rollouts | Frozen student reference $\rho$; both teachers score the generated tokens |
| Correction model | One scalar predictor on frozen reference-model features |
| Policy loss | Soft-label binary cross-entropy on student/reference chunk log-ratio differences |
| Training schedule | Build correction data, fit and freeze the predictor, then train the student |
| Deployment | Ordinary autoregressive student generation |

Cross-tokenizer transfer, new divergence families, hidden-state alignment between models, online adaptive chunking, and iterative residual bootstrapping are outside the initial method. They are not required to evaluate its central claim.

## 2. Research claim and literature position

The question is whether a simple correction makes local teacher-pair preferences useful for transferring alignment to a different student reference at an acceptable computational cost.

Two hypotheses must be tested independently:

- **Correction hypothesis:** the reference-rollout mean improves transfer over raw chunk ratios when endpoint residual differences matter and the teacher-displacement Jensen gaps vary sufficiently little among siblings.
- **Localization hypothesis:** chunk supervision offers a useful quality–cost tradeoff relative to token-level and complete-suffix supervision.

Neither hypothesis follows from the mathematical identities alone. In particular, a correction that improves a chunk target does not prove that chunks outperform complete suffixes.

The literature review through 26 October 2026 establishes the following boundaries on novelty:

| Established ingredient | Relevant primary sources | Position of this method |
|---|---|---|
| Preference-based distillation and soft targets | [PLaD](https://aclanthology.org/2024.findings-acl.923/), [PAD](https://aclanthology.org/2025.findings-acl.822/), [DP3O](https://arxiv.org/abs/2609.06893) | Soft labels and pairwise distillation are foundations. |
| Teacher advantage/value transfer | [ADPA](https://arxiv.org/abs/2502.17927), [TVKD](https://arxiv.org/abs/2509.16965) | Value-guided distillation already exists; extra computation needs justification. |
| Teacher-minus-base transfer | [Delta-MOPD](https://arxiv.org/abs/2610.10460v2), [OPD²](https://arxiv.org/abs/2607.15161) | The displacement signal itself is not new. |
| Token ratio matching and shared-prefix supervision | [TokenRatio](https://proceedings.mlr.press/v306/nguyen26l.html), [Step-DPO](https://arxiv.org/abs/2406.18629), [BranchOPD](https://arxiv.org/abs/2610.13481) | The extension concerns a specified transfer target and residual approximation. |
| Span/common-support transfer | [CTPD](https://arxiv.org/abs/2601.11865), [CSD](https://arxiv.org/abs/2610.28895) | Chunking alone is not a novelty claim. |
| Distribution-based alternatives | [Alignment as Distribution Learning](https://arxiv.org/abs/2506.01523), [daDPO](https://arxiv.org/abs/2506.15717) | Direct distribution matching is a relevant alternative to pairwise fitting. |
| Distillation failure mechanisms | [The Many Faces of OPD](https://arxiv.org/abs/2605.11182), [Mismatch Matters](https://arxiv.org/abs/2608.09836), [When EOS Tokens Disagree](https://arxiv.org/abs/2609.20511) | Reference mismatch, prefix shift, missing actions, and termination are distinct issues. |

The proposed contribution is the combination of a fixed sequence target, a practical mean correction, its explicit approximation error, and evidence about when localizing preference supervision is worthwhile. The Jensen and soft-value identities use established mathematics. Novelty of the complete contribution requires comparison with the closest methods and should not be inferred from a different name.

[Logit Grafting](https://openreview.net/pdf/c52673f0033fab5c9c8b9f53e56d06169edbeeea.pdf) reports that moderate-strength grafting changes 4–13% of generated tokens relative to its unguided baseline. This concerns changed output tokens, not a proof that log-ratio rewards are zero elsewhere, or that their full-suffix range, variance, or exponential tails are small. It motivates an alignment-displacement diagnostic; it does not establish the approximation assumptions for this method.

## 3. Notation and the fixed sequence objective

We use dimensionless log-ratios to remove redundant temperature notation. Let $s$ contain the prompt $x$ and a generated prefix, and let $y$ be a complete response. All distributions have positive support on the admissible finite response space.

Define

$$
d_T(y\mid x)=\log\frac{T^+(y\mid x)}{T^0(y\mid x)},\qquad
\delta_T(c\mid s)=\log\frac{T^+(c\mid s)}{T^0(c\mid s)},\qquad
a_\theta(c\mid s)=\log\frac{\pi_\theta(c\mid s)}{\rho(c\mid s)}.
\tag{1}
$$

The sequence objective and ideal transferred policy are

$$
J_x(\pi)=\mathbb E_\pi d_T(y\mid x)-\mathrm{KL}(\pi\|\rho),
\qquad
\pi^*(y\mid x)=\frac{\rho(y\mid x)e^{d_T(y\mid x)}}{Z(x)}.
\tag{2}
$$

The teacher log-ratio is a declared preference proxy. Its agreement with human judgment must be evaluated independently. The identity in (2) holds even if this proxy is imperfect.

In the earlier notation, $A_T=\beta\delta_T$, $\Phi=\beta\phi$, and the earlier regularized objective is $\beta J_x$. The loss scale below is $\alpha=\beta/\tau$. Thus this is the same objective at $\gamma=1$, expressed without a redundant reward-unit parameter. Introducing a new transfer exponent or using a fixed teacher reward scale different from the student KL scale changes the target and is not part of version 1.1.

A chunk is the output of a fixed causal stopping rule. The main rule stops after $L$ tokens or at EOS/horizon, whichever occurs first. The complete action set $\mathcal C_L(s)$ is prefix-free and its probabilities sum to one. For each model,

$$
\log\pi(c\mid s)=\sum_{j=1}^{|c|}\log\pi(c_j\mid s,c_{<j}).
\tag{3}
$$

Use sums of token log-probabilities. Dividing by chunk length would define a different model. The horizon convention, treatment of EOS, and scoring support must agree across all three frozen models and the student.

## 4. Exact target and the approximation we will implement

For a complete suffix $z$ after $s$, define the future alignment displacement $d_s(z)$ and reference mismatch $\ell_s(z)$. The exact residual has two equivalent representations:

$$
\begin{aligned}
d_s(z)&=\log\frac{T^+(z\mid s)}{T^0(z\mid s)},\qquad
\ell_s(z)=\log\frac{\rho(z\mid s)}{T^0(z\mid s)},\\
\phi(s)&=\log\mathbb E_{z\sim\rho(\cdot\mid s)}e^{d_s(z)}
=\log\mathbb E_{z\sim T^+(\cdot\mid s)}e^{\ell_s(z)},\\
\mu(s)&\equiv\mu_\rho(s)=\mathbb E_{z\sim\rho(\cdot\mid s)}d_s(z).
\end{aligned}
\tag{4}
$$

The exact student/reference chunk displacement is

$$
a^*(c\mid s)
=\delta_T(c\mid s)+\phi(sc)-\phi(s).
\tag{5}
$$

This is the teacher-pair decomposition developed in the user's preceding draft. Its analytic component $\delta_T$ is obtained from teacher-forced likelihoods; it requires no continuation estimation. For siblings, subtracting (5) eliminates $\phi(s)$ but leaves $\phi(sc)-\phi(sd)$.

The implemented approximation is

$$
\widehat\phi(s)=\widehat\mu_\psi(s),\qquad
t_s(c)=\delta_T(c\mid s)+\widehat\mu_\psi(sc),
\qquad \widehat\mu_\psi(y)=0\text{ at termination}.
\tag{6}
$$

It estimates an ordinary mean rather than a log-expectation of exponentials. Its error is analyzed explicitly in Section 7. It is not an unbiased estimator of $\phi$. Throughout the remaining main method, $\mu$ means $\mu_\rho$, and $\widehat\mu$ predicts that quantity.

The v1.0 alternative $\mu_{T^+}(s)=\mathbb E_{T^+}\ell_s$ remains an ablation. The two biases are exactly $\phi-\mu_\rho=\mathcal J_\rho(d_s)$ and $\phi-\mu_{T^+}=\mathcal J_{T^+}(\ell_s)$, with $\mathcal J$ defined in Section 7. Neither approximation uniformly dominates the other. The original bound was valid, but controlled the wrong perturbation for the intended small-alignment/large-base-mismatch regime.

Two exact cases should be implemented directly:

- If $T^+=T^0$ as distributions, $d_s=\phi=\mu_\rho=0$ for every prefix. No preference update is needed. The new population approximation respects this case without a special cancellation.
- If $\rho=T^0$ as distributions, $\phi=0$ and the exact transfer is $T^+$. **Bypass the predictor and set $\widehat\phi=0$.** Here $\mu_\rho(s)=-\mathrm{KL}(T^0(\cdot\mid s)\|T^+(\cdot\mid s))$ is generally nonzero, so using the new mean blindly would introduce error.

Checkpoint identity is sufficient to recognize these cases; model-family membership is not. The theorems below analyze the mean branch. The shared-reference bypass is exact by Proposition 1. Near a shared reference with large teacher displacement, the new approximation may be worse than raw ratios or the v1.0 alternative. Both neighborhoods must be evaluated.

## 5. Final policy loss

For two sibling chunks $c,d$, set

$$
\widehat p_s(c,d)=\sigma\!\left(\alpha[t_s(c)-t_s(d)]\right),\qquad
m_\theta(s,c,d)=\alpha[a_\theta(c\mid s)-a_\theta(d\mid s)],
\tag{7}
$$

where $\alpha>0$ controls preference encoding and optimization. With exact realizable targets, changing $\alpha$ does not change the recovered target policy. With finite data and limited capacity, it affects saturation and fitting behavior.

The sole policy-training objective in the main method is

$$
\boxed{
\mathcal L_{\mathrm{CRPD}}(\theta)
=\mathbb E_{s,(c,d)\sim\mathcal D}
\left[\operatorname{softplus}(m_\theta)-\widehat p_s(c,d)m_\theta\right].
}
\tag{8}
$$

The references, correction predictor, and target labels are frozen. Compute the loss in logit space. Randomize pair orientation or use the symmetric equivalent. All probabilities in (7) are teacher-induced labels; they are not direct observations of human judgments about incomplete chunks.

For a pair, the derivative is $\sigma(m_\theta)-\widehat p$, and the excess loss is Bernoulli KL. This is the logistic member of Bregman ratio matching, using $h(R)=R\log R-(1+R)\log(1+R)$ with $R=e^{-m}$. The loss is standard; the target construction and its usefulness are the research objects.

Under connected full-support comparisons, realizability, and population optimization, the loss identifies

$$
q_{\widehat\mu}(c\mid s)
=\frac{\rho(c\mid s)e^{\delta_T(c\mid s)+\widehat\mu(sc)}}
{\sum_{d\in\mathcal C_L(s)}\rho(d\mid s)e^{\delta_T(d\mid s)+\widehat\mu(sd)}}.
\tag{9}
$$

The full denominator is needed for the theoretical policy definition, not for evaluating (8). Pair differences cancel it. A finite candidate subset identifies only relative probabilities within that subset; it does not determine the probability of unobserved chunks.

Auxiliary SFT or human-label losses are not included in the primary method. If stabilization is empirically necessary, report an explicitly anchored variant and give every relevant baseline the same anchor. Such an addition changes the optimum of (8).

## 6. Training the scalar mean predictor

### 6.1 Data and targets

Choose seed prefixes from a frozen student-reference prefix pool, including a preselected subset of sibling endpoints. Draw complete continuations from $\rho(\cdot\mid s)$ to the declared EOS/horizon. Use sampling from that model distribution; uncorrected top-$p$, top-$k$, or temperature changes would change the expectation in (4).

Score each sampled continuation under $T^+$ and $T^0$ using teacher-forced forward passes. For every causally selected visited prefix $s_t$, compute

$$
Y_t=\sum_{j\ge t}
\left[\log T^+(y_j\mid s_j)-\log T^0(y_j\mid s_j)\right].
\tag{10}
$$

These labels are obtained by a reverse cumulative sum. Conditional on $s_t$, $\mathbb E[Y_t\mid s_t]=\mu(s_t)$. Each complete trajectory supplies labels at multiple prefixes, reducing repeated generation, but those labels are correlated.

Fit one scalar predictor on frozen reference-model features using squared loss. A small linear head or two-layer MLP is sufficient for the initial implementation; its architecture is a capacity choice to validate, not another claimed contribution. Enforce the terminal value to be exactly zero.

Use a predetermined prefix schedule, such as every token or every fixed number of tokens. A valid per-trajectory training loss is the sum of selected squared errors divided by the fixed maximum horizon $H$. Avoid dividing each trajectory by its realized length: that denominator depends on the future suffix and can alter the conditional regression target. Split correction training and validation by whole prompts/trajectories, not individual prefixes.

### 6.2 Freeze before policy training

Choose the predictor checkpoint on held-out correction data and freeze it. Evaluate it at each candidate endpoint to create the preference targets in (7). The main protocol uses no online target refresh or temporal-difference bootstrapping.

A head can fail through model misspecification or poor coverage of student-generated endpoints. Sampling from $\rho$ matches the frozen reference's continuation distribution, but does not guarantee coverage of teacher-proposed endpoints or a trained student that moves away from $\rho$. Include endpoint seeds explicitly and report held-out error and pair-difference error. Monte Carlo concentration for queried prefixes does not automatically establish uniform generalization of a learned head.

## 7. Main theory

The analysis concerns finite, complete chunk action spaces and the declared teacher proxy. Human utility, neural optimization convergence, and empirical speedup are separate questions.

### Proposition 1. Exact transfer and identification

Equation (5) holds for every admissible chunk rule. With exact residuals, multiplying the induced chunk policies yields (2), independently of the partition. Moreover,

$$
J_x(\pi^*)-J_x(\pi)=\mathrm{KL}(\pi\|\pi^*).
\tag{11}
$$

Connected comparisons over the full action space identify (9) in a realizable population model. They do not provide a finite-sample guarantee for unseen actions.

### Proposition 2. Exact sibling-error identity

Let $e(u)=\widehat\mu(u)-\phi(u)$ and $\mathcal J_p(f)=\log\mathbb E_p e^f-\mathbb E_p f$. Then

$$
\mathrm{KL}(\pi^*\|q_{\widehat\mu})
=\mathbb E_{\pi^*}\sum_k
\mathcal J_{\pi^*(\cdot\mid s_k)}(e(s_k\cdot)),
\qquad
\mathrm{KL}(q_{\widehat\mu}\|\pi^*)
=\mathbb E_{q_{\widehat\mu}}\sum_k
\mathcal J_{q_{\widehat\mu}(\cdot\mid s_k)}(-e(s_k\cdot)).
\tag{12}
$$

Each local KL is at most $\mathrm{osc}_c(e(sc))^2/8$. Constants shared across all sibling endpoint errors have no effect. The reverse identity, together with (11), bounds the regularized regret. It is not interchangeable with the forward identity.

### Proposition 3. First-order correction and fourth-order local KL

At a boundary $s$, suppose that for every admissible endpoint $u=sc$, the range of the future **teacher displacement** $d_u(z)$ over all complete suffixes has width at most $\varepsilon_s$. Then

$$
0\le\phi(u)-\mu(u)\le\frac{\varepsilon_s^2}{8}.
\tag{13}
$$

If $\widehat\mu=\mu$, the local KL in either direction between $q_\mu$ and $\pi^*$ obeys

$$
\boxed{\mathrm{KL}_{\mathrm{local}}\le\frac{\varepsilon_s^4}{512}.}
\tag{14}
$$

More generally, if $|\widehat\mu(u)-\mu(u)|\le\eta_s$ for every sibling endpoint, then

$$
\boxed{
J_x(\pi^*)-J_x(q_{\widehat\mu})
\le\frac18\mathbb E_{q_{\widehat\mu}}\sum_k
\left(2\eta_{s_k}+\frac{\varepsilon_{s_k}^2}{8}\right)^2.
}
\tag{15}
$$

The fourth-order statement concerns approximation error as the uniform teacher-displacement range shrinks. It is not a fourth-order statistical convergence rate, a guarantee that correction always helps, or a bound on a fitted neural student without further assumptions. Long suffixes may have large ranges; observed small variance does not certify a range bound. Boundary errors still accumulate over a response.

### Sibling-sensitive form without a small global range assumption

Define the endpoint Jensen gap $g(u)=\phi(u)-\mu(u)=\mathcal J_{\rho(\cdot\mid u)}(d_u)$ and $\omega_s=\mathrm{osc}_{c\in\mathcal C_L(s)}g(sc)$. With exact means, the local KL in either direction is at most $\omega_s^2/8$. With the same predictor error bound as above,

$$
J_x(\pi^*)-J_x(q_{\widehat\mu})
\le\frac18\mathbb E_{q_{\widehat\mu}}\sum_k(\omega_{s_k}+2\eta_{s_k})^2.
\tag{15a}
$$

This form can remain informative when each individual gap is large but the gaps are similar across siblings. It follows directly from $e=(\widehat\mu-\mu)-g$. The small-range result is the sufficient condition $\omega_s\le\varepsilon_s^2/8$, not a necessary condition for useful correction.

There is also an exact variance representation. Let $K_u(t)=\log\mathbb E_{\rho(\cdot\mid u)}e^{t d_u}$ and $\rho_{u,t}(z)=\rho(z\mid u)e^{t d_u(z)-K_u(t)}$. Then

$$
g(u)=\int_0^1(1-t)\,\mathrm{Var}_{\rho_{u,t}}(d_u)\,dt,
\qquad
\omega_s\le\int_0^1(1-t)\,
\mathrm{osc}_c\mathrm{Var}_{\rho_{sc,t}}(d_{sc})\,dt.
\tag{15b}
$$

The untilted variance $v(u)=\mathrm{Var}_{\rho(\cdot\mid u)}(d_u)$ gives the approximation $g(u)\approx v(u)/2$ only with a remainder condition. Specifically, if $|K_u'''(t)|\le C_s$ for all sibling endpoints and $t\in[0,1]$, Taylor's theorem gives

$$
|g(u)-v(u)/2|\le C_s/6,
\qquad
\omega_s\le\tfrac12\mathrm{osc}_c v(sc)+C_s/3.
\tag{15c}
$$

Here $K_u'''$ is the third centered moment under the tilted distribution. A small empirical variance under $\rho$ alone does not control it or exclude rare high-weight suffixes. Variance estimates from $Y_t$ are useful diagnostics; they are not certified estimates of $g$. Endpoint length similarity alone also does not imply similar gaps.

### Mean-estimation concentration

For $N_q$ prefixes selected before estimation, suppose each future teacher log-ratio $d_u$ has range width at most $\varepsilon$, and draw $n$ independent reference continuations per prefix. Hoeffding's inequality yields, simultaneously with probability at least $1-\delta$,

$$
|\widehat\mu(u)-\mu(u)|
\le\varepsilon\sqrt{\frac{\log(2N_q/\delta)}{2n}}.
\tag{16}
$$

There is no exponential range factor in this mean-estimation bound. It does not eliminate the need to generate data, and it applies to direct means at queried prefixes, not automatically to predictions of a function approximator. Trajectory reuse creates correlations and must not be counted as independent repeated samples at each endpoint.

## 8. Connection to TokenRatio

Let $D_T(s)$ be the teacher log-ratio accumulated along the prefix after prompt $x$. For terminal reward $d_T(y\mid x)$, define the ordinary reference-policy value of forcing a chunk $c$ and then continuing under $\rho$:

$$
Q^\rho(s,c)=\mathbb E_{z\sim\rho(\cdot\mid sc)}d_T(sc z\mid x)
=D_T(s)+\delta_T(c\mid s)+\mu(sc).
$$

Thus the **population** mean-corrected score $t_s^\mu(c)=\delta_T(c\mid s)+\mu(sc)$ is exactly $Q^\rho(s,c)-D_T(s)$. A learned head approximates this equality. This matches the reference-Q score used in [TokenRatio, Section 4.2](https://arxiv.org/html/2605.12288v3#S4.SS2), when reference, reward, and scales are matched and actions are chunks. The correction predictor estimates an ordinary continuation value; it is not TokenRatio's partition-baseline head.

The exact sequence-transfer score is instead $t_s^*(c)=\delta_T(c\mid s)+\phi(sc)=t_s^\mu(c)+g(sc)$. Hence the exact sibling log-odds differ from the population Q-based log-odds by $\alpha[g(sc)-g(sd)]$. An ordinary-Q construction is a first-order approximation to **this specified soft sequence target**. This does not reclassify all TBPO-Q objectives as approximations to an objective they were not designed to optimize.

For any local score $S(s,c)$, write $q_S(c\mid s)=\rho(c\mid s)e^{S(s,c)-b_S(s)}$. Cross-state policy/reference ratios contain $e^{b_S(s')-b_S(s)}$; sibling comparisons cancel that factor. Under the matched Q score this is the same cancellation that sets TokenRatio's state correction $w_t$ to one at a shared prefix. It does **not** remove the endpoint Jensen-gap difference.

Chunk length one therefore recovers the matched-reward, same-prefix Q-score model in the main mean branch. The training data, frozen continuation predictor, and sibling comparisons still differ from the original TBPO-Q/A algorithms. The connection is now exact at the score level, without asserting full algorithmic equivalence or newness of ordinary Q estimation.

## 9. Implementation protocol and computation

```text
Inputs:
    frozen aligned teacher T+, teacher reference T0, student reference rho;
    trainable student pi; fixed horizon H; chunk length L; pair scale alpha.

1. Construct a frozen prefix pool from training prompts and rho rollouts.
2. At selected prefixes, generate K sibling chunks under the fixed rule.
   Score chunks under T+, T0, and rho; cache their log-probabilities.
3. Select a correction-data budget and seed prefixes before collecting targets.
   Include a subset of candidate endpoints and independent rho-prefix seeds.
4. Generate complete rho continuations from those seeds.
   Score under T+ and T0; compute suffix sums Y_t for visited prefixes.
5. Fit the scalar mean predictor using the prescribed squared regression.
   Select its checkpoint on disjoint prompts/trajectories and freeze it.
6. Form t_s(c) = delta_T(c|s) + mu_hat(sc).
   Construct soft labels for sibling pairs and freeze the target table.
7. Train pi using only the soft preference objective (8).
8. Evaluate free-running responses, cost, predictor error, and candidate mass.
```

For the first implementation, use fixed-length chunks with initial $L=32$, $K=4$ candidates, and all six unordered pairs. A simple candidate design uses two independent reference samples and two teacher samples, with sampling/generation cost recorded by source. Deduplicate; if fewer than two distinct candidates remain, skip the state. Origin does not determine preference labels. These are starting engineering settings, not theory-derived optima. Use validation to select from a small predeclared grid, for example $L\in\{8,32,64\}$ and $\alpha\in\{0.1,0.3,1\}$, with equal tuning budgets for baselines. Hold test data out of selection.

Define policy sampling weights explicitly: prompts uniformly, then selected states within each prompt, then unordered pairs within a state. The target predictor is trained under the different, explicitly described trajectory-weighting scheme of Section 6. Combining their datasets does not make their sampling measures identical.

Apply the exact-case shortcuts in Section 4 before this protocol. Reference trajectories may also supply reference-proposed candidates if the reuse and prefix-selection rule are specified in advance and do not select continuations by their realized returns. Count reused tokens once.

If $P$ states have $K$ candidates of average length $\bar L$ and the correction dataset contains $M$ reference trajectories of average suffix length $\bar T$, the approximate generated-token count, excluding initial prefix collection and before reuse discounts, is

$$
C_{\rm generation}\approx PK\bar L+M\bar T.
\tag{17}
$$

Candidate scoring uses both teacher models and the student reference. Correction trajectories use $\rho$ for autoregressive generation and both $T^+$ and $T^0$ for teacher-forced scoring; full suffix tokens still pass through both teachers. This removes large-teacher autoregressive generation from correction collection, but does not remove teacher computation. Include prefix prefill, cache reuse, predictor training, student optimization, and model-dependent token costs in GPU-hour/FLOP accounting. The predictor is evaluated on frozen reference features, so many required endpoint features can be cached with reference scoring.

The benefit is amortization across prefixes and replacement of an exponential-moment target by a mean. There is no guarantee that the needed $M$ is much smaller than $PKn$ from the original approach. If predictor coverage requires a large $M$, the expected cost benefit may disappear. Report this outcome directly.

## 10. Fixed evaluation plan

### 10.1 Stage A: controlled mechanism study

Use enumerable autoregressive trees to vary teacher alignment displacement and reference mismatch independently, and compute $\phi$, both means, and $\pi^*$ exactly. Sweep chunk length and predictor error. Check exact partition invariance, Jensen-gap identities, the Q-score identity, and both the sibling-gap and fourth-order bounds. Include cases where coarsening worsens approximation and where either mean correction is worse than zero correction.

Include exact and near-zero teacher-displacement controls, especially $T^+=T^0$ with a different $\rho$, as well as exact and near-shared-reference cases $\rho\approx T^0$. Test the exact shortcuts separately from the unmodified approximations. A valid counterexample narrows a claim; it is not to be removed from the evaluation suite.

The reproducible revision checks in Appendix D cover five regimes, six seeds, and four chunk lengths. They confirm the algebra and demonstrate the two approximation regimes. They are not measurements of language-model quality, practical sample efficiency, or novelty.

### 10.2 Stage B: decisive language-model comparisons

Use at least two student capacities, one verified aligned/reference teacher pair, a general preference task, and a reasoning task with independently verifiable correctness. Record exact checkpoints and preprocessing; do not assume a public base checkpoint is the reference used for the teacher's alignment. Separate correction training, correction validation, policy training, hyperparameter selection, and final evaluation as required to prevent test leakage.

The mandatory comparisons are fixed:

| Method | Role |
|---|---|
| Raw sibling chunk ratios, $\widehat\phi=0$ | Tests whether the correction adds value |
| ChunkRatio-PD with the reference-rollout mean predictor | Primary method |
| v1.0 teacher-rollout mean, $\mu_{T^+}$ | Tests approximation direction at matched total compute |
| Expensive residual estimation on a diagnostic subset | Measures approximation quality; report uncertainty of this estimator too |
| Token-sized counterpart | Tests the contribution of chunk granularity |
| Complete-suffix preference distillation from the same branching prefixes | Tests whether local supervision is worthwhile |
| Competitive teacher-delta baseline, such as adapted Delta-MOPD, plus ADPA or TVKD | Tests against existing transfer methods |
| TBPO-Q/A under their original formulation | Establishes continuity with TokenRatio |

For complete suffixes $z,z'$ from one prefix, terminal residuals are zero. The exact preference target under (2) is $\sigma(\alpha[\delta_T(z\mid s)-\delta_T(z'\mid s)])$. Apply the student margin to the complete suffixes. This baseline requires generating longer candidates but no endpoint residual model. It must receive an equal overall teacher-compute budget.

A diagnostic exponential estimator is not an oracle unless its error is controlled. Exact targets are available only in the tractable experiments; for language models, use replication and increasing sample budgets to assess stability.

### Preregistered secondary diagnostics and ablations

On fixed pilot prefixes, compare the distributions and sample variances of $d_u$ under $\rho$ and $\ell_u$ under $T^+$. Also audit sibling gap differences where credible exponential-moment estimates are feasible. Smaller observed variance is a heuristic, not a guarantee of smaller bias or better policy transfer; a large gap common to siblings can be harmless. Keep $\mu_\rho$ as the main branch and report the alternative rather than silently switching after seeing test outcomes.

A secondary second-order target may use $\widehat\mu_\rho+\widehat v_\rho/2$. On tractable trees use exact moments. In neural experiments, a variance head requires disjoint or cross-fitted mean predictions: the regression target $(Y-\widehat\mu)^2$ has conditional expectation $v+(\mu-\widehat\mu)^2$, not generally $v$. Report this contamination and higher-cumulant bias. A variance term is not guaranteed to improve the approximation and is not part of the primary method.

Uncertainty-based label tempering, decision-point branching, and prefixes from a separately trained student remain optional, preregistered ablations. They do not change the default loss, fixed chunk rule, or frozen-reference rollout measure. Prefixes from another policy still require $\rho$ continuations for unbiased labels of $\mu_\rho$. None is required for the primary claim.

### 10.3 Outcomes and analysis

Choose the primary generation-quality outcome and compute budget before final training. Report raw and length-controlled preference results, independent judge evaluation with a blinded human subset, correctness, response length, EOS behavior, diversity, safety regressions, and retained general capability. Use at least three training seeds where feasible and prompt-level paired intervals. Keep training-teacher agreement separate from independent quality.

For the correction model, report held-out suffix-return prediction error, repeated-rollout estimates on fixed audit prefixes, and errors in sibling differences. Distinguish predictor error from first-order approximation bias. Report sibling variation of estimated Jensen gaps, with uncertainty where estimable, rather than relying only on their absolute size. Repeated labels can estimate noise; they do not prove a uniform bound over all possible suffixes or actions.

For coverage, measure the total probability of the observed prefix-free candidate set under the student/reference, compare candidate sources, and evaluate free generation. A low pairwise training loss cannot substitute for these checks. EOS equivalence and stopping conventions should be validated explicitly because termination changes can confound quality and efficiency.

Run both matched-candidate comparisons and matched-total-compute comparisons. The former isolates the target; the latter measures practical utility. Include correction-data collection and hyperparameter search in the latter.

### 10.4 Decision rules after evaluation

| Evidence | Research conclusion |
|---|---|
| Correction improves independent generation at equal compute, with corresponding target-error reduction | Supports the correction hypothesis |
| Chunk supervision also improves the quality–cost tradeoff over token and complete-suffix baselines | Supports the localization hypothesis |
| Correction helps only in a restricted mismatch regime | Report that regime and its diagnostics |
| Raw ratios perform equally well or better | This implementation has not established a benefit; distinguish gap-approximation bias, predictor error, coverage, and computation. Do not infer that the exact residual is unnecessary. |
| Complete suffixes dominate at equal compute | The localization claim is unsupported |
| Gains occur only on cached comparisons or teacher agreement | Free-generation benefit is unestablished |

Failure of a hypothesis should first change the claim, not trigger an unbounded search for additional losses, heads, or segmentation rules. Any subsequent methodological change should cite the specific failed assumption or measured failure it addresses.

## 11. Publication assessment and stable claim

The current deliverable is an implementation-ready research specification, not a submission-ready journal article. It is logically more focused than a collection of optional corrections: one objective, one target approximation, one predictor, and one policy loss.

If the experiments support both hypotheses, the central claim can be:

> We characterize how reference mismatch affects chunk-level preference transfer, introduce an amortized mean correction with an explicit approximation bound, and demonstrate when localized supervision provides a better quality–cost tradeoff than token and complete-suffix alternatives.

The fourth-order local bound is conditional on a bounded teacher-displacement range and accurate mean prediction. The sibling-gap form can be sharper, but estimating its terms is itself nontrivial. These proofs combine familiar inequalities with the existing boundary calculus. A top-tier contribution must show that this analysis leads to a useful method or a substantial empirical insight. Neither theorem count nor a renamed loss establishes significance.

[JMLR's reviewer criteria](https://jmlr.org/reviewer-guide.html) ask for technical correctness, significance, sufficient distinction from prior work, and appropriate evidence. [TMLR's acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) emphasize convincing support for claims and findings of interest; a new SOTA score or methodological novelty is not independently required. No acceptance probability is defensible before the results and venue are known.

For an extension of TokenRatio, identify reused results and the new teacher-transfer problem explicitly, preserve the distinction between score models, and follow the target journal's actual prior-publication policy. The analysis and method here do not require asserting equivalence to all TBPO variants.

## Appendix A. Proofs

### A.1 Exact decomposition, objective identity, and recovery

Factor the teacher sequence ratio at a prefix. Marginalizing (2) over its completions gives a soft value whose prefix-dependent part is the teacher prefix log-ratio plus

$$
\log\sum_z\rho(z\mid s)\frac{T^+(z\mid s)}{T^0(z\mid s)}
=\log\mathbb E_{T^+(\cdot\mid s)}e^{\ell_s(z)}=\phi(s).
$$

Dividing the prefix probabilities at $sc$ and $s$ proves (5). Summing the conditional log-ratios along a response telescopes the residual to $-\phi(x)$, since $\phi(y)=0$, and recovers (2). Also, $\log(\pi^*/\rho)=d_T-\log Z$. Taking expectations under $\pi$ gives $\mathrm{KL}(\pi\|\pi^*)=\log Z-J_x(\pi)$; setting $\pi=\pi^*$ proves (11).

For the loss, each exact pair optimum has $a_\theta(c)-a_\theta(d)=t_s(c)-t_s(d)$. Connectivity over the full action space implies $a_\theta(c)=t_s(c)+k_s$. Normalization fixes $k_s$, giving (9). Finite sampled subsets leave their total probability mass unidentified.

### A.2 Jensen-gap boundary identities

Let $q=q_{\widehat\mu}$. Dividing (9) by the exact chunk policy gives

$$
q(c\mid s)=\frac{\pi^*(c\mid s)e^{e(sc)}}{\mathbb E_{\pi^*(\cdot\mid s)}e^{e(s\cdot)}}.
$$

Taking the expected log-ratio under $\pi^*$ gives the forward local Jensen gap. Expressing $\pi^*$ as a tilt of $q$ by $-e$ gives the reverse gap. Apply the KL chain rule to the finite chunk tree to obtain (12). Hoeffding's lemma gives $\mathcal J_p(f)\le\mathrm{osc}(f)^2/8$.

### A.3 Mean approximation

The two expressions in (4) agree term by term because $\rho(z\mid u)T^+(z\mid u)/T^0(z\mid u)=T^+(z\mid u)\rho(z\mid u)/T^0(z\mid u)$. By definition, $\phi(u)-\mu(u)=\mathcal J_{\rho(\cdot\mid u)}(d_u)$. Jensen's inequality gives nonnegativity, and Hoeffding's lemma gives the upper bound in (13). With exact means, $e(u)$ lies in $[-\varepsilon_s^2/8,0]$ for every sibling endpoint, so its sibling oscillation is at most $\varepsilon_s^2/8$. Applying A.2 proves (14).

With predictor error at most $\eta_s$, $e(u)$ lies in $[-\eta_s-\varepsilon_s^2/8,\eta_s]$. Its range has width at most $2\eta_s+\varepsilon_s^2/8$. The reverse identity and (11) prove (15). Applying Hoeffding's inequality to the ordinary sample mean and a union bound across $N_q$ queried prefixes proves (16).

For (15a), $\mathrm{osc}(e)\le\mathrm{osc}(g)+2\eta_s$. For (15b), $K_u(0)=0$, $K_u'(0)=\mu(u)$, and $K_u''(t)=\mathrm{Var}_{\rho_{u,t}}(d_u)$. Taylor's integral formula gives $K_u(1)-K_u'(0)=\int_0^1(1-t)K_u''(t)dt$. The oscillation of the integral is at most the integral of the oscillation. A second-order Taylor expansion gives remainder at most $C_s/6$; its difference between two endpoints is at most $C_s/3$, proving (15c). Finite support ensures the derivatives exist. These identities do not supply an empirical tail guarantee.

## Appendix B. Distinguishing target error from student fitting error

The preceding policy bounds concern the normalized target $q_{\widehat\mu}$. They do not assume that a neural student exactly equals this target. For any positive student policy define

$$
\Delta_s(c)=\log\frac{\pi_\theta(c\mid s)}{q_{\widehat\mu}(c\mid s)}.
$$

Then the exact reverse-KL identity becomes

$$
\mathrm{KL}(\pi_\theta\|\pi^*)
=\mathbb E_{\pi_\theta}\sum_k
\mathcal J_{\pi_\theta(\cdot\mid s_k)}
\big(-e(s_k\cdot)-\Delta_{s_k}(\cdot)\big).
\tag{18}
$$

Thus the relevant error is the sibling variation of the sum of residual and fitting energies. There is no unjustified triangle inequality between KL divergences.

For a connected comparison graph over the full action set, let $\mathcal R_s$ be its maximum effective resistance with edge weights summing to one. Let $\mathcal E_s$ be the edge-weighted excess soft preference loss. If both supplied and student edge logits lie in $[-M,M]$, set $\kappa_M=\sigma(M)[1-\sigma(M)]$. Logistic curvature and the Laplacian resistance inequality give

$$
\mathrm{osc}_c\Delta_s(c)
\le\frac1\alpha\sqrt{\frac{2\mathcal R_s\mathcal E_s}{\kappa_M}}.
\tag{19}
$$

To see this, the edge logit discrepancy is $\alpha(\Delta_s(c)-\Delta_s(d))$. Strong convexity on the stated interval gives $\mathcal E_s\ge\kappa_M\alpha^2\Delta_s^\top L_s\Delta_s/2$. Effective resistance bounds each squared pair difference by $\mathcal R_s\Delta_s^\top L_s\Delta_s$. Combining (18)–(19) with Proposition 3 gives an additional fitting term in the regret bound.

These are full-support and bounded-logit statements. A small empirical candidate graph or clipped logits do not automatically satisfy their assumptions. Graph resistance is an established tool; see [Hendrickx et al.](https://proceedings.mlr.press/v97/hendrickx19a.html). This appendix retains the useful fitting analysis from the preceding version without making graph machinery part of the implementation.

## Appendix C. Provenance and change record

The teacher-pair decomposition and sibling Jensen-gap calculus come from the user's ChunkRatio-PD/RADD drafts and the preceding joint review. The previous revised proposal added explicit support qualifications, both KL directions, and a fitting-error analysis. Those foundations are retained.

Version 1.0 chose the teacher expectation of the reference-mismatch log-ratio. The supplied feedback correctly identified a failure in the intended small-displacement regime. Version 1.1 replaces that approximation by $\mu_\rho$, switches correction rollouts and regression labels accordingly, and retains the old direction as an ablation. The objective, analytic chunk signal, one-head architecture, frozen training schedule, and policy loss are unchanged. This is an evidence-backed correction to the estimator, not a replacement research problem.

The revision also adds the exact Q-score connection, sibling Jensen-gap analysis with a qualified variance approximation, both identity-case controls, and a more careful interpretation of failed experiments. The feedback's quantitative language-model claims and numerical table are not assumed to be established results; Appendix D distinguishes independent checks from those supplied claims.

The independent ingredients have the following status:

- **Derived:** exact transfer identities, normalized sibling target, Jensen-gap error calculus, mean-approximation bound, and conditional fitting bound.
- **Numerically checked:** both residual representations, exact policy recovery, both KL chain identities, the Q-score identity, the tilted-variance identity, and representative cases of the revised mean-correction bounds; see Appendix D.
- **Unverified empirical claims:** effective amortization, useful predictor accuracy on student endpoints, lower total training cost, stronger free generation, and an advantage of chunks over tokens or complete suffixes.
- **Outside the guarantee:** teacher preference correctness, rare unseen actions, generalization from observed prefixes, and universal improvement under stronger mismatch or longer responses.

Version 1.1 is the stable specification for the next implementation and evaluation cycle. Evidence may narrow its scope or refute a hypothesis; it should not be silently replaced by another proposal in response to ordinary follow-up questions.

## Appendix D. Feedback assessment and reproducible verification

| Feedback claim | Assessment and action |
|---|---|
| The exact residual admits both expectation measures | Correct; shown in (4) and proved in A.3. |
| The old expansion is poorly suited to small alignment displacement with large base mismatch | Correct as a regime-specific criticism. Its theorem was not algebraically false; the new expansion is not universally better. |
| The new mean is a reference-policy Q continuation | Correct at the population score level, with matched terminal reward/reference and an additive prefix constant. Section 8 states the exact scope. |
| The 4–13% statistic proves small alignment log-ratios | Unsupported inference: it measures output-token changes in the cited experiments, not the range or tails needed here. |
| Reference rollouts are cheaper and improve coverage | Plausible operational advantages, not established total-cost or trained-student coverage guarantees. Both teachers still score all correction tokens. |
| Half the variance approximates the Jensen gap | Correct only with higher-order control; (15b) is exact, while (15c) states a sufficient remainder condition. |
| Select the approximation with smaller pilot variance | Useful diagnostic, not a reliable ordering theorem. No automatic switch is added to the primary method. |
| A raw-ratio win means residual correction is unnecessary | Too strong; the evaluation rules now distinguish estimator failure from absence of a useful exact correction. |

The supplied numerical table did not include code, distribution construction, or seed values. Its precise numbers are therefore not claimed as reproduced. An independent check uses a full ternary tree of depth five, six fixed seeds, and chunk lengths $1,2,3,5$. At each nonterminal state, base logits are sampled independently from $\mathcal N(0,0.6^2)$. Independent unit-Gaussian vectors scaled by $(a,b)$ perturb the base logits to construct $T^+$ and $\rho$, respectively. Means and suffix sums are enumerated exactly; no predictor is fitted. The table reports mean $\mathrm{KL}(q\|\pi^*)$ for one-token chunks:

| Regime $(a,b)$ | Raw | v1.0 teacher mean | v1.1 reference mean | Reference mean + half variance |
|---|---:|---:|---:|---:|
| Small displacement, large mismatch $(0.08,1)$ | 0.002191 | 0.098270 | 0.00000886 | 0.0000000143 |
| Large displacement, small mismatch $(1,0.08)$ | 0.001863 | 0.00000685 | 0.079754 | 0.023096 |
| Both moderate $(0.5,0.5)$ | 0.022191 | 0.008954 | 0.009355 | 0.000574 |
| Zero teacher displacement $(0,1)$ | 0 | 0.096525 | 0 | 0 |
| Shared reference $(1,0)$ | 0 | 0 | 0.082890 | 0.023554 |

**Identity shortcuts are deliberately disabled in this table to expose approximation behavior.** The actual method skips updating in the zero-displacement case and uses the exact zero residual in the shared-reference case. The table supports the direction-specific critique while refuting universal dominance. The second-order column is an exact-moment diagnostic, not validation of a second neural head or a guaranteed improvement.

All tested algebraic identities agree to within $1.8\times10^{-15}$. Numerical quadrature of the tilted-variance identity agrees to within $1.3\times10^{-12}$. The checked Jensen-gap, sibling-oscillation, and fourth-order inequalities hold across the 120 regime/seed/chunk configurations. Such finite checks supplement the proofs; they are not proofs of generalization or language-model effectiveness.

Reproduce with `python outputs/ChunkRatio_PD_Revision_Checks.py` from the workspace root. The script and JSON results are saved alongside this proposal as `ChunkRatio_PD_Revision_Checks.py` and `ChunkRatio_PD_Revision_Checks.json`.
