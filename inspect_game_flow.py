
import inspect
from ChefsHatGym.env.ChefsHatEnv import ChefsHatEnv

methods = [
    "getObservation",
    "step",
    "isMatchover",
    "update_player_score_roles",
    "get_current_score",
    "get_acumulated_score",
    "get_acumulated_performance_score",
]

for name in methods:
    method = getattr(ChefsHatEnv, name)
    print("\n" + "=" * 70)
    print(f"METHOD: {name}{inspect.signature(method)}")
    print(inspect.getsource(method))
