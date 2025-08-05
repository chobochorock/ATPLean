import Mathlib
def hello := "world"

universe u

def id₃ (α : Type u) (a : α) := a

#check fun x y => if not y then x + 1 else x + 2

def F (α : Type u) : Type u := Prod α α

#eval (λ x y: Nat => x + y + 5) ((λ x: Nat => x + 7) 10) 12

#check (λ x y: Nat => x + y + 5)

#eval (fun x => 2 * x) 10

#eval (fun x => 2 * x)  (-1 * 2.2)

#check F
-- #check F p


example : ∀ m n : Nat, Even n → Even (m * n) := by
  rintro m n ⟨k, hk⟩; use m * k; rw [hk]; ring


example : ∀ m n : Nat, Even n → Even (m * n) := by
    intros; simp [*,]
--   intros; fllkhalsdfkh
-- hello 

-- read_file --
-- read_start --
inductive MyTree where
    | leaf : MyTree
    | branch : List MyTree → MyTree-- (t1 : MyTree) (t2 : MyTree)

open MyTree


def num_of_vertex : MyTree → ℕ
 --    | MyTree.none => 0
    | MyTree.leaf => 1
    | MyTree.branch l => List.foldr (fun b acc ↦ acc + num_of_vertex b) 0 l + 1
--    | MyTree.branch t1 t2 => num_of_vertex t1 + num_of_vertex t2 + 1

def num_of_edge : MyTree → ℕ
    | MyTree.leaf => 0
    | MyTree.branch l => List.foldr (fun b acc ↦ acc + 1 + num_of_edge b) 0 l

#eval num_of_vertex (branch [leaf, branch [leaf]]) -- by open MyTree
#eval num_of_edge (branch [leaf, branch [leaf]])

theorem vertex_eq_edge_plus_one (t : MyTree) :
    (num_of_vertex t = num_of_edge t + 1) := match t with
    | leaf => by simp [num_of_vertex, num_of_edge]
    | branch [] => by simp [num_of_vertex, num_of_edge]
    | branch (x :: xs) => by
        simp [num_of_vertex, num_of_edge]
        have h_x := vertex_eq_edge_plus_one x
        have h_xs := vertex_eq_edge_plus_one (branch xs)
        simp [num_of_vertex, num_of_edge] at h_xs
        rw [h_x, h_xs]
        ring
-- read_end --
