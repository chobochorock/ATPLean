from lean_interact import LeanREPLConfig, LeanServer, ProofStep, Command
from lean_interact.interface import LeanError
from rl_node import RLNode


def tacticSelector():
    available_tactics = ["rw []", "intro h", "exact h"]
    return available_tactics


class Interacting:
    def __init__(self):
        config = LeanREPLConfig(verbose=True)
        server = LeanServer(config)
        self.problem = problem


# first_step = server.run(ProofStep(tactic="rw []", proof_state=0))
# # secnd_step = server.run(ProofStep(tactic="intro h", proof_state=1))
# third_step = server.run(ProofStep(tactic="exact h", proof_state=2))
# print(first_step, type(first_step), isinstance(first_step, LeanError))
# # print(secnd_step)
# print(third_step, type(third_step), isinstance(third_step, LeanError))

# ProofStepResponse(proof_state=1, goals=['n : Nat\n⊢ n = 5 → n = 5'], proof_status='Incomplete: open goals remain')
# LeanError(message='Unknown proof state.')
