# Comment 16: Original Upstream Position-Offset Injection

## Published source
VeReMi NextGen v1 (2026), Hermann et al.
Dataset DOI: https://doi.org/10.5281/zenodo.19665762

Attack archive:
InTAS_highway_2_constantPositionOffset.zip
Verified MD5: 4B8AA6B684A093DE218256BAE74C1A6E

## Injection implementation
The upstream VeReMi NextGen attackGenerator.py implements
constant_position_offset. It applies vehicle-specific X/Y
offsets and sets the attacker label.

Source:
https://github.com/VeReMi-dataset/VeReMi-NextGen

Referenced code snapshot:
acda994bae9966df6ae2f292a2945d8d91e249ce

The upstream file is preserved without modification in
upstream_veremi_nextgen/attackGenerator.py.

## Independent verification
All 99,937 NVBD reception records were matched against
the official constant-position-offset attack archive.

Unmatched records: 0
Sender-position mismatches: 0
Attacker-label mismatches: 0
Maximum position difference: 5.820766091346741e-11

Thus NVBD preserved the already-injected upstream sender
positions and attacker labels.

## Reproducibility limitation
The referenced upstream code snapshot is publicly available,
but the historical random seed and exact environment used
to produce the archived attack realization are not established.

The recovered-offset replay is a separate verification
utility, not the original upstream injection generator.

## Licensing and redistribution
The VeReMi NextGen generator is Apache-2.0 licensed.
The generator's LICENSE and NOTICE files are retained.

The released VeReMi NextGen dataset specifies CC BY 4.0,
permitting redistribution and adaptation with attribution
and other applicable licence conditions.

The underlying InTAS traffic scenario is separately
GPL-3.0 licensed. Its original scenario files are not
included in this repository update.

These licences and their respective conditions must
be respected independently.
