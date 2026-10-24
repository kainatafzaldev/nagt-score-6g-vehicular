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
