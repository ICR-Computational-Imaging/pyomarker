<p align="center">
  <img src="https://github.com/ICR-Computational-Imaging/pyomarker/blob/main/docs/assets/logo.png?raw=true" alt="Pyomarker logo" width="400"/>
</p>

# pyomarker

[![PyPI version](https://img.shields.io/pypi/v/pyomarker?color=blue)](https://pypi.org/project/pyomarker/)
[![License](https://img.shields.io/pypi/l/pyomarker)](https://github.com/ICR-Computational-Imaging/pyomarker/blob/main/LICENSE.md)
[![Tests](https://github.com/ICR-Computational-Imaging/pyomarker/actions/workflows/test_publish.yml/badge.svg)](
https://github.com/ICR-Computational-Imaging/pyomarker/actions/workflows/test_publish.yml
)
[![Docs](https://github.com/ICR-Computational-Imaging/pyomarker/actions/workflows/deploy_docs.yml/badge.svg)](
https://github.com/ICR-Computational-Imaging/pyomarker/actions/workflows/deploy_docs.yml
)

**Bayesian and classical methods for quantitative imaging biomarker reliability and uncertainty.**

Quantitative imaging biomarkers are increasingly used in clinical research, yet their repeatability
and uncertainty are often poorly characterised. **pyomarker** provides well-defined frequentist and
Bayesian tools for analysing test–retest data, estimating measurement reliability, and quantifying
uncertainty in imaging biomarker studies. It leverages Hamiltonian Monte Carlo sampling of parameter
posterior distributions through the [Stan probabilistic programming framework](https://mc-stan.org/).

---

> ⚠️ **Development status**  
> pyomarker is under active development. APIs may change.
> If you would like more information about upcoming releases, please [contact me](mailto:matthew.blackledge@icr.ac.uk).

---

### Key features
- Classical repeatability metrics (Bland–Altman, ICC, CoV)
- Bayesian models for uncertainty-aware biomarker analysis
- Designed for quantitative imaging workflows

## Installation

```bash
pip install pyomarker
```

📘 **Documentation:** [https://icr-computational-imaging.github.io/pyomarker/](https://icr-computational-imaging.github.io/pyomarker/)


## Example

```python
import numpy as np
from pyomarker.models.test_retest.real.bland_altman import BlandAltman

x1 = np.array([1.2, 1.4, 1.1, 1.3])
x2 = np.array([1.3, 1.5, 1.0, 1.2])

ba = BlandAltman(ci=0.90).fit(x1, x2)
print(ba.metrics())
```

Check our [examples page](https://github.com/ICR-Computational-Imaging/pyomarker/tree/main/examples) for more.