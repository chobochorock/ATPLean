from lean_dojo import *

# print("hello world")
# /home/chorock/.cache/lean_dojo/leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5.tar.gz
repo = LeanGitRepo(
    "https://github.com/leanprover-community/mathlib4",
    "29dcec074de168ac2bf835a77ef68bbe069194c5",
)
# theorem = Theorem(repo, file_path=)
trace_repo = trace(repo)

theorem = Theorem(trace_repo, "Mathlib/Algebra/BigOperators/Pi.lean", "pi_eq_sum_univ")
print(theorem)

dojo, state_0 = Dojo(theorem).__enter__()
print(dojo, state_0)
