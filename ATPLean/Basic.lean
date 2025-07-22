import Mathlib
-- import Mathlib.inductive

def hello := "world"

universe u

def id₃ (α : Type u) (a : α) := a

#check fun x y => if not y then x + 1 else x + 2

def F (α : Type u) : Type u := Prod α α

#eval (λ x y: Nat => x + y + 5) ((λ x: Nat => x + 7) 10) 12

#check (λ x y: Nat => x + y + 5)

#eval (fun x => 2 * x) 10

#check F
-- #check F p


example : ∀ m n : Nat, Even n → Even (m * n) := by
  rintro m n ⟨k, hk⟩; use m * k; rw [hk]; ring

example : ∀ m n : Nat, Even n → Even (m * n) := by
    intros; simp [*,]
--   intros; simp [*, parity_simps]



inductive BiTree where
    | none : BiTree
    | node : BiTree
    | children : BiTree → BiTree → BiTree-- (t1 : BiTree) (t2 : BiTree)

open BiTree

def is_not_none : BiTree → ℕ
    | BiTree.none => 0
    | _ => 1
    -- | BiTree.node => 1
    -- | BiTree.children t1 t2 => 1

def num_of_vertex : BiTree → ℕ
    | BiTree.none => 0
    | BiTree.node => 1
    | BiTree.children t1 t2 => num_of_vertex t1 + num_of_vertex t2 + 1

def num_of_edge : BiTree → ℕ
    | BiTree.none => 0
    | BiTree.node => 0
    | BiTree.children t1 t2 => num_of_edge t1 + num_of_edge t2+ is_not_none t1 + is_not_none t2

#eval num_of_vertex (children node (children none node)) -- by open BiTree
#eval num_of_edge (BiTree.children BiTree.node (BiTree.children BiTree.none BiTree.node))

theorem vertex_eq_edge_plus_one : ∀t : BiTree,
    (t ≠ none) → (num_of_vertex t = num_of_edge t + 1) := by
    intro t
    induction t with
        -- | none => sorry
        -- | node => sorry
        -- | children t1 t2 ih1 ih2 => sorry
        | none =>
            -- BiTree.none의 경우
            simp
        | node =>
            -- BiTree.node의 경우
            simp [num_of_vertex, num_of_edge]
        | children t1 t2 ih1 ih2 =>
            -- 귀납 가정:
            -- ih1 : num_of_vertex t1 = num_of_edge t1 + 1
            -- ih2 : num_of_vertex t2 = num_of_edge t2 + 1
            simp [num_of_vertex, num_of_edge, is_not_none]
            cases t1 <;> cases t2 <;> simp [is_not_none] at *
            -- 모든 경우의 수를 나눠서 계산
            repeat
                rw [ih1, ih2]
                decide
    -- induction' n with n ih
