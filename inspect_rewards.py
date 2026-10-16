
import inspect
from ChefsHatGym.env.ChefsHatEnv import ChefsHatEnv

methods = [
    "calculateScore",
    "get_current_score",
    "get_acumulated_score",
    "get_acumulated_performance_score",
    "update_player_score_roles",
    "isMatchover",
]

for name in methods:
    method = getattr(ChefsHatEnv, name)
    print(f"\n{'=' * 60}")
    print(f"METHOD: {name}{inspect.signature(method)}")
    print(inspect.getsource(method))
