-- def main : IO Unit := IO.println "Hello, world!"

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout

  stdout.putStrLn "How would you like to be addressed?"
  let input ← stdin.getLine
  let name := input.dropRightWhile Char.isWhitespace
  stdout.putStrLn s!"Hello, {name}!"


def expandRewrite (stx : Syntax) : MacroM Syntax := do
  match stx with
  | `(tactic| rw [$args,*]) => 
    let rwTactics ← args.getElems.mapM fun arg =>
      `(tactic| rw [$arg])
    `(tactic| $rwTactics:tactic*)
  | _ => return stx

def expandSimpRw (stx : Syntax) : MacroM Syntax := do
  match stx with
  | `(tactic| simp_rw [$args,*]) =>
    -- simp_rw를 simp + rw 시퀀스로 변환
    ...


def expandByTerm (stx : Syntax) : MetaM Syntax := do
  match stx with
  | `(by $tac) => 
    -- tactic을 더 explicit한 형태로 변환
    let expanded ← expandTactic tac
    `(by $expanded)

partial def transformTactics (stx : Syntax) : MacroM Syntax := do
  match stx with
  | .node info kind args => 
    let newArgs ← args.mapM transformTactics
    return .node info kind newArgs
  | .atom _ _ => return stx
  | .ident _ _ => return stx

def expandTactic (stx : Syntax) : MacroM Syntax := do
  match stx with
  | `(tactic| rw [$args,*]) => expandRewrite stx
  | `(tactic| simp_rw $args*) => expandSimpRw stx
  | `(tactic| simp [$args,*]) => expandSimp stx
  | _ => transformTactics stx  -- 재귀적으로 하위 구조 처리

