# CRPD configurations

`smoke.json` is resolved and runs on CPU without downloads. Each named variant
JSON and `protocol.json` fixes the revision-2 settings but deliberately leaves
unknown checkpoint paths/revisions null. Validation rejects those unresolved
research inputs. Fill actual teacher DPO/SFT provenance and student SFT revisions,
then set `teacher_dpo_reference_verified` and pin the selection reward-model revision.

The method scale is alpha=.3, separately from teacher DPO beta=.05. CRPD optimizer,
learning rate, batch, steps and correction budget are explicit starting choices
(DECISIONS.md), not literature-validated optima. Main uses prompts only; only
crpd_anchor consumes chosen_ids/rejected_ids. Primary weight decay is zero.

Canonical JSONL rows contain `prompt_ids`, `split`, and optionally `group_id`,
original `messages` (required for real reward selection), `chosen_ids`,
`rejected_ids` (required for anchor/preparation). Split duplicate groups before
any generation. `prepare-data` serializes local exported chat pairs or HelpSteer
helpfulness records with the pinned shared tokenizer; it rejects changing prompt
tokenization at the response seam. Raw Deita SFT exports must be converted to
the same canonical format with chosen_ids as the supervised answer.

The sft_teacher/sft_student/dpo_teacher JSONs have the exact preparation recipes
and require actual model revisions/data. Their single-process full fine-tuning
launcher is unvalidated on large checkpoints. It does not replace an official
baseline implementation. The teacher-DPO manifest records its exact reference
fingerprint. Full-scale training is intentionally not run by smoke.

```text
python -m chunkratio prepare-data --input data/export.jsonl --tokenizer MODEL --revision COMMIT --output data/canonical.jsonl
python -m chunkratio train-sft --config config/crpd/sft_teacher.json
python -m chunkratio train-sft --config config/crpd/sft_student.json
python -m chunkratio train-teacher-dpo --config config/crpd/dpo_teacher.json
python -m chunkratio validate-config --config config/crpd/protocol.json
python script/run_crpd_protocol.py --config config/crpd/protocol.json --dpo-prompts data/dpo_mix_7k.jsonl --helpsteer-prompts data/helpsteer2.jsonl
```

The last command only creates run configs; `--execute` runs both tracks and
three seeds. Windows/Linux wrappers are `script/run_crpd_protocol.ps1` and `.sh`.
See REPORT.md for verification, external benchmark prerequisites and limitations.
