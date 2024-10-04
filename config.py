import __future__
from dataclasses import dataclass 
from typing import List

@dataclass
class TGWL_Bayesian:
    
    model_type: str = ["nnt", "mcmc_TGWL", "nnt_real"]
    params: str = ["alpha", "gamma","epsilon","tau"]
    burn_in: int = 1000
    num_iterations: int = 5000
    samples: List[int] = [200, 1000, 10000]
    
    
