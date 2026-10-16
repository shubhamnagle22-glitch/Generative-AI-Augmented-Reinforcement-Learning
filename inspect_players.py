
import inspect
from ChefsHatGym.env.ChefsHatEnv import ChefsHatEnv

methods = [
    "start_match_handle_cards",
    "start_match_define_order",
    "getPossibleActions",
    "getRandomAction",
    "player_finished",
]

for name in methods:
    method = getattr(ChefsHatEnv, name)
    print(f"\n{'=' * 60}")
    print(f"METHOD: {name}{inspect.signature(method)}")
    print(inspect.getsource(method))
