# Comment 16: Source-Position Reproducibility Audit

## Purpose

This directory provides independently verified
source-position auditing and replay materials for
the NAGT-V2X Benchmark Dataset (NVBD).

## Source Dataset

VeReMi NextGen, InTAS Highway 2
DOI: https://doi.org/10.5281/zenodo.19665762

Source archive: InTAS_highway_2.zip
Verified MD5:
D5FEFB873C5E766B702BD036F18A326D

The upstream dataset specifies CC BY 4.0.
See UPSTREAM_DATA_ATTRIBUTION.md for attribution
and adaptation details.

## Verified Upstream Attack Provenance

The position offsets in NVBD were already present in
the official VeReMi NextGen constant-position-offset
attack archive. They were not newly injected during
construction of the NVBD CSV.

Official attack archive:
InTAS_highway_2_constantPositionOffset.zip

Verified archive MD5:
4B8AA6B684A093DE218256BAE74C1A6E

Independent comparison against this archive:
- NVBD reception records: 99,937
- Matched source records: 99,937
- Unmatched records: 0
- Sender-position mismatches: 0
- Attacker-label mismatches: 0
- Maximum position difference: 5.820766091346741e-11

The clean InTAS_highway_2.zip archive was used
separately to investigate the original attack offsets.
The recovered-offset manifest and replay utilities
remain independent retrospective verification tools.

## Authentic Upstream Injection Implementation

The VeReMi NextGen attackGenerator.py contains the
constant_position_offset attack implementation.

Upstream repository:
https://github.com/VeReMi-dataset/VeReMi-NextGen

Referenced upstream commit:
acda994bae9966df6ae2f292a2945d8d91e249ce

The upstream implementation snapshot is located at:
upstream_veremi_nextgen/attackGenerator.py

Its Apache-2.0 LICENSE and NOTICE files are retained.
The released VeReMi NextGen dataset is CC BY 4.0,
permitting redistribution subject to attribution
and the other applicable licence terms.

See UPSTREAM_INJECTION_PROVENANCE.md for details.

The original random seed and exact historical
execution environment are not established. The
referenced code snapshot is not claimed to be
the exact historical version that generated the
published archive.

## Verification Results

Original reception records matched: 99,937
Attacker sender/split offset groups: 221
Legitimate positions modified: 0
Nonconstant attacker offsets: 0
Replay position mismatches: 0
Attacker-label mismatches: 0

Maximum absolute replay position error:
1.45519152284e-10

Both source comparison and independent replay PASSED
at the specified position tolerance of 1e-5.

## Files

NAGT_Comment16_Source_Offset_Audit_FIXED.py
    Source-to-NVBD comparison utility.

NAGT_Comment16_Replay_Offsets.py
    Replays the recovered attacker position offsets.

inferred_sender_offsets.csv
    Offset manifest recovered from NVBD.

construction_audit.txt
    Source-comparison verification report.

replay_report.txt
    Independent replay verification report.

UPSTREAM_DATA_ATTRIBUTION.md
    Dataset source, licence and attribution notes.

## Reproduction

Obtain the NVBD dataset.csv and the matching
original InTAS Highway 2 Train, Validation
and Test receiver-log archives.

Run the source audit script using:
--dataset
--train-source
--validation-source
--test-source
--outdir

Then run the replay script using the same
dataset and source archives, together with
--manifest inferred_sender_offsets.csv.

The full command-line options are available
through each script's --help argument.

## Important Limitations

These are independently reconstructed verification
utilities, not the original injection-generation
program.

The offsets were recovered from the published NVBD.
Therefore, successful replay does not independently
verify the original random-number generator, its seed,
or the complete 52-column construction pipeline.

The replay demonstrates source-relative reproducibility
of sender positions and attacker labels. It does not
establish generalization to other attack scenarios.
