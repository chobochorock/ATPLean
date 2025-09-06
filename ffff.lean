import Lean
import Lean.Meta
import Lean.Parser

open Lean Meta Elab Parser

-- 분석 결과를 담을 구조체 (이전 코드와 동일)
structure ReversibleDeclarationInfo where
  name : Name
  kind : String
  nameSpace : Name
  typeExpr : String
  valueExpr : String
  deriving Repr

structure FileAnalysisStats where
  fileName : String
  totalDeclarations : Nat
  definitionCount : Nat
  theoremCount : Nat
  deriving Repr

-- 선언의 종류를 문자열로 가져오는 간단한 도우미 함수
def getDeclarationKind (info : ConstantInfo) : String :=
  match info with
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

-- Expr를 문자열로 예쁘게 변환하는 함수
def exprToString (e : Expr) : MetaM String := do
  return (← prettyPrinter e).pretty

-- 특정 모듈에 정의된 선언들만 가져오는 함수
def getFileDeclarations (moduleName : Name) : MetaM (Array Name) := do
  let env ← getEnv
  let some modIdx := env.getModuleIdx? moduleName
    | return #[] -- 모듈을 찾지 못하면 빈 배열 반환
  let constants := env.constants.map₂.toArray
  let mut decls := #[]
  for (name, _) in constants do
    if env.getModuleIdxFor? name == some modIdx then
      decls := decls.push name
  return decls

-- 단일 선언을 분석하는 함수
def analyzeDeclarationReversible (name : Name) : MetaM (Option ReversibleDeclarationInfo) := do
  let env ← getEnv
  match env.find? name with
  | none => return none
  | some info => do
    -- 사용자 정의 선언이 아니거나, 내부적으로 생성된 이름은 건너뜁니다.
    if name.isInternal || !(← Meta.isUserDefinedConstant name) then
      return none

    let kind := getDeclarationKind info
    let valueExpr ← match info with
      | .defnInfo val | .thmInfo val => exprToString val.value
      | _ => pure ""

    return some {
      name := name,
      kind := kind,
      nameSpace := name.getPrefix,
      typeExpr := (← exprToString info.type),
      valueExpr := valueExpr
    }

-- 파일 전체를 분석하는 메인 함수
def analyzeFile (filePath : String) : MetaM (Array ReversibleDeclarationInfo × FileAnalysisStats) := do
  -- 파일 경로를 모듈 이름으로 변환 (예: "Mathlib/Data/Nat/Basic.lean" -> `Mathlib.Data.Nat.Basic)
  let moduleName := Lean.Name.mkSimple <| filePath.replace ".lean" "".replace "/" "."
  let declarations ← getFileDeclarations moduleName

  IO.println s!"🔍 {moduleName} 모듈에서 {declarations.size}개의 선언을 발견했습니다. 분석을 시작합니다..."

  let mut results : Array ReversibleDeclarationInfo := #[]
  let mut defCount := 0
  let mut thmCount := 0

  for name in declarations do
    if let some info ← analyzeDeclarationReversible name then
      results := results.push info
      match info.kind with
      | "def" => defCount := defCount + 1
      | "theorem" => thmCount := thmCount + 1
      | _ => pure ()

  let stats : FileAnalysisStats := {
    fileName := filePath,
    totalDeclarations := results.size,
    definitionCount := defCount,
    theoremCount := thmCount
  }
  return (results, stats)
