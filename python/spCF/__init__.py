"""spCF — Coarse-to-fine spatial modeling (Python).

Python port of the R package `spCF <https://cran.r-project.org/package=spCF>`_.
"""
from .cf_lm import CFLM, CFLMHV, cf_lm, cf_lm_hv
from .cf_glm import CFGLM, CFGLMHV, cf_glm, cf_glm_hv
from .sp_scalewise import sp_scalewise
from . import families

__version__ = "0.1.1"
__all__ = [
    "cf_lm", "cf_lm_hv", "CFLM", "CFLMHV",
    "cf_glm", "cf_glm_hv", "CFGLM", "CFGLMHV",
    "sp_scalewise",
    "families",
]
