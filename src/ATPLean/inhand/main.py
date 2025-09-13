import lean_interact_exer


if __name__ == "__main__":
    problem = server.run(
        Command(cmd="theorem ex (n : Nat) : n = 5 → n = 5 := sorry")
        # Command(cmd="theorem mathd_numbertheory_328 : (5^999999) % 7 = 6 := sorry ")
    )
    goal = [elem.goal for elem in problem.sorries]
    print(goal)
    root = RLNode(goal=goal)
    current_node = root
    proof_queue = [(current_node, [tactic]) for tactic in tacticSelector()]
    while goal != []:
        current_node, tactics = proof_queue.pop(0)
        current_goal = current_node.goal
        problem = server.run(
            Command(cmd="theorem ex (n : Nat) : n = 5 → n = 5 := sorry")
            # Command(cmd="theorem mathd_numbertheory_328 : (5^999999) % 7 = 6 := sorry ")
        )

        proof_step = None
        for i, tactic in enumerate(tactics):
            proof_step = server.run(ProofStep(tactic=tactic, proof_state=i))
            if isinstance(proof_step, LeanError):
                proof_step = None
                current_node.add_child(RLNode(goal="error", used_tactic=tactics[-1]))
                break

        if proof_step is not None:
            new_node = current_node.add_child(
                RLNode(goal=proof_step.goals, used_tactic=tactics[-1])
            )
            for tactic in tacticSelector():
                proof_queue.append((new_node, tactics.copy() + [tactic]))
            # TODO:
            # new_node.evaluating_tactics()

            index = new_node.index
            result_goal = new_node.goal
            print(index, result_goal)
