# PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions

## Overview

This project migrates the PyTorch implementation of the paper *PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions* to the MindSpore framework, providing a new loss function for classification tasks.

## Run

```bash
python -m unittest -v test_polyloss.py 
```

## Output

```bash
test1 (test_polyloss.TestPolyBCELoss) ... ok
test2 (test_polyloss.TestPolyBCELoss) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.110s

OK
```

## References

- Paper: [PolyLoss Paper](https://paperswithcode.com/paper/polyloss-a-polynomial-expansion-perspective)
- Original Code: [PyTorch Implementation](https://github.com/yiyixuxu/polyloss-pytorch)