# GE1 Frozen Stage 0 Configuration and Preregistration

## Record status

| Field | Frozen value |
|---|---|
| Protocol identity | `GE1-STAGE0-PREREG-v1` |
| Experiment identity | `GE1-SHARED-DECODER-ENCODER-COMPARISON` |
| Status | C0 complete; protocol frozen; Stage 1 not begun |
| Frozen on | August 7, 2026 |
| Authority | accepted [ADR-0004](../decisions/ADR-0004-ge1-single-manifest-encoder-comparison.md) |
| Designated GE1 reviewer | Krishay Maskara |
| Model implementation begun | `false` |

This record freezes the Stage 0 scientific choices. It does not establish
that a model, checkpoint, or experiment result exists. A change to a frozen
choice requires a superseding ADR. Filling the manifest hash and verified
physical counts below is completion of a prescribed metadata check, not a
scientific protocol change.

## Model, decoder, and checkpoint identity

| Component | Frozen identity/version |
|---|---|
| Model family/schema | `GE1-MODEL-v1` |
| Flat control arm | `GE1-FLAT-CHRONOLOGICAL-CONTINUOUS-V1` |
| Typed-graph treatment arm | `GE1-TYPED-GRAPH-POSITION-FREE-CONTINUOUS-V1` |
| Shared decoder | `GE1-SHARED-TYPED-EDGE-DECODER-V1` |
| Checkpoint schema | `GE1-CHECKPOINT-v1` |
| Selected checkpoint | epoch 50 for every arm and retained seed |
| Filename pattern | `{arm}-seed{seed}-epoch0050.pt` |

Each checkpoint must record the protocol identity, experiment identity, arm
identity, decoder identity, checkpoint schema, seed, epoch, manifest SHA-256,
configuration hash, code revision, and parameter inventory. Strict reload
must reject a mismatch. Training loss, plateau epoch, development results,
and protected results cannot select a checkpoint.

The initial shared decoder is the Graph V1 independent ordered-pair typed-edge
main path, trained from scratch in each arm. It retains the existing
output-side source and destination decoder-position embeddings and signed
relative output position because those locate predicted output slots; it does
not include C1's additive directed position-bias branch. These output-side
signals are shared identically between arms and are not graph-encoder inputs.
The graph encoder receives no absolute or chronological position embedding,
`operation_sequence`, or tensor index as a semantic feature. Local node
indices, edge endpoints, `graph_offsets`, graph IDs, masks, padding positions,
and output-alignment indices are bookkeeping only.

Exactly one common decoder correction is preauthorized before comparative
development inspection if train-only sufficiency fails: the hierarchical
operation-group decoder specified by ADR-0004. No Graph V1 C2 is authorized.

## Manifest authority and metadata audit

The sole authorized manifest is the physical file:

```text
/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026/manifests/operation_template.json
```

| Metadata field | Recorded value |
|---|---|
| Authoritative physical path identified from durable corpus record | `true` |
| Metadata-only audit source | local SHA-pinned copy, `/Users/krishaymaskara/Downloads/operation_template.json` |
| Verified `train` families / samples | 407 / 814 |
| Verified `validation` families / samples | 45 / 90 |
| Verified `test` families / samples | 114 / 228 |
| Verified `secondary_systematic_validation` families / samples | 114 / 228 |
| Verified total families / samples | 680 / 1,360 |
| Unique family IDs / sample IDs | 680 / 1,360 |
| Samples per family | exactly 2 |
| Orphan samples / partition mismatches | 0 / 0 |
| Counts verified on the local SHA-pinned copy | `true` |
| Runtime binding to the Adroit file | pending; enforced at first Stage 1 read |
| Authoritative SHA-256 | `a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7` |

Two distinct claims are recorded separately, because the remote metadata
connection failed during C0 and the physical Adroit file was never opened.

**Verified now.** The audit ran against a local copy, reproduced the SHA-256
above, and confirmed the declared and recomputed 407/45/114/114 family counts,
1,360 unique sample records, exactly two samples per family, and no orphan or
partition-mismatched sample metadata. These values agree with the checked-in
seed-2026 split contract and accepted ADR-0004. No referenced CAD history
payload was opened.

**Not yet verified.** That the file at the authoritative Adroit path is
byte-identical to the audited local copy. Nothing in C0 establishes this. The
binding is deferred to runtime: `prototype.graph_encoder.partitions` hashes the
physical manifest on every load and raises `manifest_authority_failure` on any
mismatch, so the first Stage 1 read is what converts the local audit into a
statement about the authoritative file. Until that read succeeds, every count
and assignment hash in this section is a property of the local copy alone.

Assignment hashes use SHA-256 over the UTF-8 bytes of lexicographically sorted
`source_family_id` values, one per line with a final LF:

| Assignment | SHA-256 |
|---|---|
| `train` | `42d61d2224ae1a2110279f66d374516f8b5f220271407913147d8be390c66595` |
| `validation` | `9af48e34e0ae2e5fd266d2336b6adecdad543c75a2c470ea132b9c7a593afaf6` |
| `test` | `b18df0bdf9cc95575cc79f6cd2bdded5f9559e0964305a525969a2702de22663` |
| `secondary_systematic_validation` | `eb37468f4baf6540891add9293a66aee2077ecd7cb64d7cb486902a6eb58c494` |

These four values received a reviewer-authorized factual correction on August
7, 2026, after a metadata-only audit found that the earlier values had been
computed with literal backslash-plus-`n` delimiters. The correction preserves
the already documented actual-LF algorithm and changes no manifest bytes,
assignment, partition policy, experimental protocol, or access rule.

Only `operation_template.train` and `operation_template.validation` payloads
may be loaded after Stage 1 begins. RR may be opened once at Stage 7. ER remains
closed throughout GE1. No partition from any other manifest is authorized.

The C1 loader `prototype.graph_encoder.partitions` refuses every protected
partition unconditionally and carries no authorization argument. The Stage 7 RR
evaluation must therefore add a **new, explicitly named, separately audited
entry point**. Relaxing, parameterizing, or adding an override to the existing
C1 loader is prohibited: protected access must remain visible in the call graph
and reviewable on its own, and the train/development path must stay incapable
of reaching protected data by configuration.

### Stage 0 payload-access ledger

The audit in this documentation session used repository documentation,
filenames, archive member names, split-generation logic, and a failed
metadata-only remote connection. It opened zero physical CAD history payloads.

| Protected source | Payload records opened in Stage 0 | Access state |
|---|---:|---|
| Operation-template RR (`secondary_systematic_validation`) | 0 | closed until Stage 7 |
| Operation-template ER (`test`) | 0 | closed throughout GE1 |
| IID, every partition | 0 | unauthorized throughout GE1 |
| History-depth, every partition | 0 | unauthorized throughout GE1 |
| Geometry-extrapolation, every partition | 0 | unauthorized throughout GE1 |

Manifest assignment metadata is not a payload. Once connectivity exists, the
authorized verification may inspect only manifest bytes, assignment keys,
family identifiers needed to count/hash assignments, file size, and hashes.

## Frozen training and capacity configuration

| Field | Frozen value |
|---|---|
| Planned seeds | `2026`, `2027`, `2028` |
| Authorized timing fallback | `2026`, `2027` only |
| Epochs | 50 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | 0 |
| Gradient clipping | `1.0` |
| Expected steps per epoch | 51 |
| Steps per arm and seed | 2,550 |
| Example presentations per arm and seed | 20,350 |
| Planned total | 15,300 steps across six runs |
| Two-seed fallback total | 10,200 steps across four runs |
| Parameter-count tolerance | at most 5% total trainable-count difference |
| Allowed capacity adjustments | relation-basis count or encoder feed-forward width only |

The August 7 reviewer acceptance replaced checkpoint selection only; it did
not require three seeds unconditionally. The two-seed fallback therefore
remains authorized. It may be invoked only after one train-only timing epoch
per arm on intended hardware, with projected wall time, peak memory, storage,
and 20% contingency recorded before full training. If invoked, both retained
seed-level treatment effects must be nonnegative. Epochs may not be reduced;
if two 50-epoch seeds per arm are infeasible, GE1 is blocked.

Train-loss plateau is less than 1% relative improvement in the five-epoch
moving best after epoch 10. Failure to plateau by epoch 50 makes the comparison
optimization-inconclusive but does not change checkpoint selection. The same
classification applies if arm train-ceiling shortfalls differ by more than
0.05 for normalized executable prefix or 0.10 for complete validity.

## Prospective C7-v2 and Stage 6 budget addendum

The tables above remain the immutable `GE1-STAGE0-PREREG-v1` and C7-v1
history. Accepted
[ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md) prospectively
supersedes only their 50-epoch budget and checkpoint semantics for C7-v2 and
for eventual Stage 6 if C7-v2 later authorizes it. It does not edit the
historical C7-v1 selection.

| Prospective use | Epochs | Fixed checkpoint | Updates | Presentations |
|---|---:|---:|---:|---:|
| C7-v2 tiny, four families | 200 | epoch 200 | 200 per arm | 800 per arm |
| C7-v2 scaled, 32 families | 200 | epoch 200 | 800 per arm | 6,400 per arm |
| Eventual Stage 6, 407 families | 200 | epoch 200 | approximately 10,200 per arm/seed | exactly 81,400 per arm/seed |

Every architecture, optimizer, loss, cohort, seed, endpoint, memory threshold,
capacity, and access rule remains unchanged. C7-v2 and eventual Stage 6 have
no early stopping, best-checkpoint selection, extension after observation,
warm start, C7-v1 resume, or optimization-diagnostic checkpoint reuse.

Stage 6 remains blocked unless a finalized C7-v2 artifact explicitly records
`stage6_authorized_by_c7_v2=true`. The C7-v1 decoder-repair trigger remains in
history but is deferred. C7-v2 exact failure returns to that repair path;
memory-only failure remains inconclusive and does not trigger repair.

## Frozen sufficiency, endpoints, and decisions

Train-only sufficiency uses one E, R, EE, and RE family, then a deterministic
32-family set with eight families from each template. The autonomous
memory-use gate requires nonzero true-memory normalized prefix and both
shuffled/true and mean/true ratios at most 0.80 in each arm.

The powered primary endpoint is the paired-family mean treatment-minus-control
change in autonomous normalized longest executable operation prefix on the 45
operation-template development families. It must be at least 0.10 after first
computing a family mean within each retained seed and then averaging seeds.
Development data cannot select checkpoints or hyperparameters.

### Development composition and conditional subset sensitivity

The 45 development families are recorded here so that the endpoint's resolution
is fixed before any result exists:

| Template | Operations | Families |
|---|---:|---:|
| `E` | 1 | 10 |
| `R` | 1 | 7 |
| `EE` | 2 | 13 |
| `RE` | 2 | 15 |
| Single-operation subtotal | 1 | 17 |
| Two-operation subtotal | 2 | 28 |
| Total | — | 45 |

These counts were recomputed from the SHA-pinned manifest copy and inherit the
runtime-binding caveat above.

Single-operation families admit scores of `0` or `1`; two-operation families
admit `0`, `0.5`, or `1`. Frozen Graph V1 solved every single-operation IID
family, so single-operation families may exhibit little headroom. This is a
**conditional sensitivity note, not an assumption**: no prior constrains the
single-operation treatment effect, and it is not assumed to be zero.

Stated conditionally: *if* the single-operation treatment effect is exactly
zero, the overall 0.10 threshold requires a mean improvement of approximately
`0.161` on the two-operation subset, since `0.10 x 45 / 28 = 0.160714...`. A
nonzero single-operation effect in either direction changes the required
two-operation improvement accordingly.

The 0.10 threshold applies to the full 45-family mean and is unchanged by this
note. Results must additionally be reported stratified by operation count so
that the realized composition of the effect is visible rather than inferred.

The one-time confirmatory endpoint is the same measure on 114 RR families,
requiring all of:

- retained-seed-mean improvement of at least 0.10;
- at least 25% of paired RR families advance one operation on average; and
- no more than 5% regress one operation on average.

If RR normalized-prefix seed means tie exactly, apply the frozen ordered
seven-stage first-failure comparison in the implementation plan. Results are
classified as support, mixed, null, or inconclusive using ADR-0004's frozen
rules. ER and all other manifests remain unopened regardless of outcome.

## C0 completion ledger

| Requirement | State |
|---|---|
| ADR accepted and reviewer named | complete |
| Model, decoder, and checkpoint identities assigned | complete |
| Fixed epoch-50 checkpoint selection synchronized | complete |
| Seeds and authorized fallback settled | complete |
| Capacity, optimization, and plateau rules frozen | complete |
| Endpoints, effect rules, tie-break, and correction budget frozen | complete |
| Zero protected-payload access recorded | complete |
| Authoritative physical manifest path identified | complete |
| Physical 407/45/114/114 counts and hashes verified | complete |
| C0 activation gate | complete |
| Stage 1 | not begun by reviewer instruction |
| Prospective ADR-0008 budget addendum | accepted and recorded August 10, 2026 |

C0 is complete. No model implementation or scientific training began during
this audit, and Stage 1 remains deliberately unstarted.
