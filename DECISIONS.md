# Decisions

1. **Precedence**: explicit task > coding spec > method > executable checks >
   inherited conventions, as requested. The spec itself gives mathematics to the
   method; no mathematical conflict has been found.
2. **Reference filenames**: supplied `ChunkRatio_Math_Checks.*` are the actual
   executable sources; spec's Revision_Checks names are aliases from an earlier
   delivery. Preserve supplied files and hash them rather than invent fixtures.
3. **Reference numbers**: Appendix D's historical full ternary-tree table differs
   from the supplied variable-EOS script. Match the executable JSON fixtures;
   retain early EOS and report the discrepancy, not a changed research claim.
4. **Environment**: active interpreter is Python 3.15.1. Initially no torch,
   numpy, transformers or legacy training stack was installed. Stable PyPI has
   no compatible torch wheel; the official CPU nightly index supplies a cp315
   wheel. Install that into the same active interpreter (no interpreter switch).
   Record the exact nightly build and actual executed results in REPORT.md;
   initial dependency-blocked attempts are not successful smoke runs.
5. **Baseline marker**: `tbpo-baseline` was absent. Created it at the starting
   HEAD before shared changes. Source comparisons supplement (do not replace)
   unavailable runtime baseline equivalence checks.
6. **Integration**: opt-in `train.py --crpd ...` dispatch before legacy imports
   avoids the inherited Unix-only resource/FSDP/dependency initialization on
   Windows. CRPD owns its offline batch construction and loop; legacy paths stay
   unchanged. Distributed CRPD is not claimed.
7. **Research inputs**: exact checkpoint revisions, split data, judge versions,
   hardware and throughput are unresolved inputs. Fail closed for missing input;
   never fabricate benchmark scores or a measured GPU-hour estimate.
8. **Eligible measure**: prompts/states with fewer than two distinct candidates
   are recorded as skipped and excluded from policy sampling. Normalize over
   eligible prompts, then their eligible states, then all unordered pairs.
9. **Engineering settings**: the spec fixes H=512, L=32, K=4, the preparation
   recipes and weight decay=0. It does not fix CRPD alpha/lr/steps. Configs use
   alpha=.3, AdamW lr=1e-6, effective batch=128, microbatch=1, 1000 policy steps,
   20 head epochs at lr=.001 and four correction replicates. These are declared
   starting settings, not validated optima or values borrowed from teacher beta.
10. **Sampling resume**: independent semantic job seeds plus saved before/after
    RNG state make a partially written job restart identically. Completed shard
    records are immutable; interrupted uncommitted work is recomputed from its
    original seed, never from an outcome-dependent retry.
11. **Audit variant**: crpd_audit fits the main head and reports independent
    exponential-moment estimates with bootstrap uncertainty. It is diagnostic,
    takes zero policy optimizer steps, and exports the initialized student.
12. **External comparisons**: official ADPA/TVKD/DPKD/DCKD/SimPO implementations
    are not present. Do not substitute CRPD variants. Benchmark CLI plans require
    pinned external installations and official generation/annotator configs;
    generated plans are reviewable templates, not verified benchmark results.
