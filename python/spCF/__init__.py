"""spCF — Coarse-to-fine spatial modeling (Python).

Python port of the R package `spCF <https://cran.r-project.org/package=spCF>`_.
"""
from .cf_lm import CFLM, CFLMHV, cf_lm, cf_lm_hv
from .cf_glm import CFGLM, CFGLMHV, cf_glm, cf_glm_hv
from .cf_downscale import CFDownscale, CFDownscaleHV, cf_downscale, cf_downscale_hv
from .cf_dglm import CFDGLM, CFDGLMHV, cf_dglm, cf_dglm_hv
from .sp_scalewise import sp_scalewise
from . import families

__version__ = "0.1.3"
__all__ = [
    "cf_lm", "cf_lm_hv", "CFLM", "CFLMHV",
    "cf_glm", "cf_glm_hv", "CFGLM", "CFGLMHV",
    "cf_downscale", "cf_downscale_hv", "CFDownscale", "CFDownscaleHV",
    "cf_dglm", "cf_dglm_hv", "CFDGLM", "CFDGLMHV",
    "sp_scalewise",
    "families",
]
