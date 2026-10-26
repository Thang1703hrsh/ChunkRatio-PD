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
4. **Environment**: active interpreter is Python 3.15.1. No torch, numpy,
   transformers or legacy training stack was installed. PyPI has no compatible
   torch distribution for this interpreter. Do not switch interpreters (explicit
   task constraint). Tensor checks must skip with a clear dependency reason;
   smoke must fail clearly, never report synthetic success without execution.
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
