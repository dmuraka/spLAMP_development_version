"""GLM family helpers.

Thin wrapper around statsmodels family objects, exposing the subset of
attributes used by the R `family()` API: linkfun, linkinv, mu_eta, plus
a label for the link name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import statsmodels.api as sm
from statsmodels.genmod import families as smf


@dataclass
class Family:
    """Lightweight wrapper around a statsmodels family object."""
    family: smf.Family
    name: str
    link_name: str

    def linkfun(self, mu):
        return self.family.link(np.asarray(mu, dtype=float))

    def linkinv(self, eta):
        return self.family.link.inverse(np.asarray(eta, dtype=float))

    def mu_eta(self, eta):
        return self.family.link.inverse_deriv(np.asarray(eta, dtype=float))

    def variance(self, mu):
        return self.family.variance(np.asarray(mu, dtype=float))

    @property
    def link(self) -> str:
        return self.link_name


def gaussian(link: Optional[str] = None) -> Family:
    link = link or "identity"
    fam = smf.Gaussian(link=_make_link(link))
    return Family(family=fam, name="gaussian", link_name=link)


def poisson(link: Optional[str] = None) -> Family:
    link = link or "log"
    fam = smf.Poisson(link=_make_link(link))
    return Family(family=fam, name="poisson", link_name=link)


def binomial(link: Optional[str] = None) -> Family:
    link = link or "logit"
    fam = smf.Binomial(link=_make_link(link))
    return Family(family=fam, name="binomial", link_name=link)


def gamma(link: Optional[str] = None) -> Family:
    link = link or "inverse"
    fam = smf.Gamma(link=_make_link(link))
    return Family(family=fam, name="gamma", link_name=link)


def _make_link(name: str):
    name = name.lower()
    mapping = {
        "identity": sm.genmod.families.links.Identity(),
        "log": sm.genmod.families.links.Log(),
        "logit": sm.genmod.families.links.Logit(),
        "probit": sm.genmod.families.links.Probit(),
        "cloglog": sm.genmod.families.links.CLogLog(),
        "inverse": sm.genmod.families.links.InversePower(),
        "sqrt": sm.genmod.families.links.Sqrt(),
    }
    if name not in mapping:
        raise ValueError(f"unsupported link: {name!r}")
    return mapping[name]


def as_family(obj) -> Family:
    """Coerce input to a Family. Accepts Family, statsmodels family, or a string."""
    if isinstance(obj, Family):
        return obj
    if isinstance(obj, smf.Family):
        link_name = type(obj.link).__name__.lower().replace("identity", "identity")
        # Best-effort extraction; default link names match statsmodels.
        link_label = {
            "identity": "identity",
            "log": "log",
            "logit": "logit",
            "probit": "probit",
            "cloglog": "cloglog",
            "inversepower": "inverse",
            "sqrt": "sqrt",
        }.get(link_name, link_name)
        return Family(family=obj, name=type(obj).__name__.lower(), link_name=link_label)
    if isinstance(obj, str):
        return {"gaussian": gaussian, "poisson": poisson, "binomial": binomial, "gamma": gamma}[obj.lower()]()
    raise TypeError(f"cannot interpret {obj!r} as Family")
