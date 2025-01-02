# SMOTE: Synthetic Minority Over-sampling Technique

## Overview

This project migrates the PyTorch implementation of the paper *SMOTE: Synthetic Minority Over-sampling Technique* to the MindSpore framework, generating synthetic samples for minority classes.

## Run

```bash
python smote_mindspore.py
```

## Output

```bash
Original dataset shape: (120, 2)
Balanced dataset shape: (200, 2)
Class distribution after balancing:
{0: 100, 1: 100}
```

## References

- Paper: [SMOTE Paper](https://paperswithcode.com/paper/smote-synthetic-minority-over-sampling)
- Original Code: [PyTorch Implementation](https://github.com/example/smote)