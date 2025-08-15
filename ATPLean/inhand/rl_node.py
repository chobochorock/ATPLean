class RLNode:
    def __init__(
        self,
        goal: str,
        index: str = "root",
        parent=None,
        used_tactic: str = "",
    ):
        self.goal = goal
        self.index = index
        self.used_tactic = used_tactic
        self.children = []
        self.parent = parent
        # consider add weight or...

    def add_child(self, goal, tactic):
        index = (
            self.index + f".{len(self.children) + 1}"
            if self.index != "root"
            else f"{len(self.children) + 1}"
        )
        new_node = RLNode(goal, index, self, tactic)
        self.children.append(new_node)
        return new_node
