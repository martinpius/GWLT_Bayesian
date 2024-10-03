import __future__
from dataclasses import dataclass 

@dataclass
class TGWL_Bayesian:
    model_type: str = ["nnt", "mcmc_TGWL", "nnt_real"]