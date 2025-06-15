import numpy as np

def create_version_dictionary(UC_pairs):
    UC_dict = dict()
    for v,UC_pair in enumerate(UC_pairs):
        UC_m = UC_pair[0]
        UC_cov = UC_pair[1]
        UC_dict.update({v:{
                            "UC_mean": UC_m,
                            "UC_cov": UC_cov
                            }
                        })

    for UC in UC_dict.keys():
        print(f"v{UC} - UC_mean: {UC_dict.get(UC).get("UC_mean")} - UC_cov {UC_dict.get(UC).get("UC_cov")}")
    
    return UC_dict