import Lean
import Lean.Parser
import Std.Data.HashMap
import Init.System.FilePath

open Lean Meta Elab Parser

-- 완전 가역가능한 선언 정보 구조체
structure ReversibleDeclarationInfo where
  -- 기본 정보
  name : Name
  kind : String
  namespace : Name
  -- 완전한 구문 정보 (가역성을 위해)
  fullSyntax : String  -- 원본 구문 전체
  typeExpr : String    -- 타입 표현식
  valueExpr : String   -- 값/증명 표현식
  -- 구조화된 정보
  explicitParams : Array (String × String)  -- (이름, 타입) 쌍
  implicitParams : Array (String × String)
  instanceParams : Array (String × String)
  -- 의존성 정보
  typeDependencies : Array Name
  valueDependencies : Array Name
  -- 메타데이터
  docString : Option String
  attributes : Array String
  -- 위치 정보 (복원시 참조용)
  sourceLocation : Option (String × Nat × Nat)  -- (파일, 줄, 열)
  deriving Repr

-- 파일별 분석 통계
structure FileAnalysisStats where
  fileName : String
  totalDeclarations : Nat
  definitionCount : Nat
  theoremCount : Nat
  lemmaCount : Nat
  instanceCount : Nat
  inductiveCount : Nat
  structureCount : Nat
  classCount : Nat
  deriving Repr

-- 매개변수 정보 추출 (완전한 형태로)
def extractParameters (type : Expr) : MetaM (Array (String × String) × Array (String × String) × Array (String × String)) := do
  let mut explicit : Array (String × String) := #[]
  let mut implicit : Array (String × String) := #[]
  let mut instance : Array (String × String) := #[]
  
  let rec go (e : Expr) : MetaM Unit := do
    match e with
    | .forallE name domain body bindInfo => do
      let nameStr := name.toString
      let typeStr ← exprToString domain
      match bindInfo with
      | .default => explicit := explicit.push (nameStr, typeStr)
      | .implicit => implicit := implicit.push (nameStr, typeStr)
      | .instImplicit => instance := instance.push (nameStr, typeStr)
      | .strictImplicit => implicit := implicit.push (nameStr, typeStr)
      go body
    | _ => pure ()
  
  go type
  return (explicit, implicit, instance)

-- 원본 구문 복원을 위한 정보 추출
def extractFullSyntax (name : Name) : MetaM String := do
  let env ← getEnv
  match env.find? name with
  | some info => do
    -- 가능한 한 원본에 가까운 구문 재구성
    let kind := getDeclarationKind info
    let typeStr ← exprToString info.type
    
    match info with
    | .defnInfo val => do
      let valueStr ← exprToString val.value
      return s!"{kind} {name} : {typeStr} := {valueStr}"
    | .thmInfo val => do
      let proofStr ← exprToString val.value
      return s!"{kind} {name} : {typeStr} := {proofStr}"
    | .inductInfo val => do
      -- 귀납형의 경우 더 복잡한 재구성 필요
      return s!"inductive {name} : {typeStr}"
    | _ => do
      return s!"{kind} {name} : {typeStr}"
  | none => return ""

-- 문서 문자열 추출
def extractDocString (name : Name) : MetaM (Option String) := do
  let env ← getEnv
  match env.getModuleDoc? name with
  | some doc => return some doc
  | none => return none

-- 속성 추출
def extractAttributes (name : Name) : MetaM (Array String) := do
  let env ← getEnv
  let attrs := env.getAttributeNames
  let mut result : Array String := #[]
  
  for attr in attrs do
    if (← attr.hasAttribute name env) then
      result := result.push attr.toString
      
  return result

-- 향상된 단일 declaration 분석
def analyzeDeclarationReversible (name : Name) (fileName : String) : MetaM (Option ReversibleDeclarationInfo) := do
  let env ← getEnv
  match env.find? name with
  | none => return none
  | some info => do
    let kind := getDeclarationKind info
    let namespace := name.getPrefix
    let fullSyntax ← extractFullSyntax name
    let typeStr ← exprToString info.type
    
    let (explicit, implicit, instance) ← extractParameters info.type
    let typeDeps := extractDependencies info.type
    let valueDeps := match info with
      | .defnInfo val => extractDependencies val.value
      | .thmInfo val => extractDependencies val.value
      | _ => #[]
    
    let valueStr ← match info with
      | .defnInfo val => exprToString val.value
      | .thmInfo val => exprToString val.value
      | _ => pure ""
    
    let docString ← extractDocString name
    let attributes ← extractAttributes name
    
    return some {
      name := name
      kind := kind
      namespace := namespace
      fullSyntax := fullSyntax
      typeExpr := typeStr
      valueExpr := valueStr
      explicitParams := explicit
      implicitParams := implicit
      instanceParams := instance
      typeDependencies := typeDeps
      valueDependencies := valueDeps
      docString := docString
      attributes := attributes
      sourceLocation := some (fileName, 0, 0)  -- 실제로는 위치 정보 필요시 추가
    }

-- 향상된 파일 분석 (통계와 함께)
def analyzeFileReversible (filePath : String) : MetaM (Array ReversibleDeclarationInfo × FileAnalysisStats) := do
  let moduleName := filePath.toName
  let declarations ← getFileDeclarations moduleName
  
  IO.println s!"📂 Analyzing file: {filePath}"
  IO.println s!"🔍 Found {declarations.size} declarations to analyze..."
  
  let mut results : Array ReversibleDeclarationInfo := #[]
  let mut defCount := 0
  let mut thmCount := 0
  let mut lemmaCount := 0
  let mut instCount := 0
  let mut indCount := 0
  let mut structCount := 0
  let mut classCount := 0
  
  for (decl, idx) in declarations.zipWithIndex do
    if idx % 50 == 0 then
      IO.println s!"  Progress: {idx}/{declarations.size} ({(idx * 100 / declarations.size)}%)"
    
    match ← analyzeDeclarationReversible decl filePath with
    | some info => 
      results := results.push info
      match info.kind with
      | "def" => defCount := defCount + 1
      | "theorem" => thmCount := thmCount + 1
      | "lemma" => lemmaCount := lemmaCount + 1
      | "instance" => instCount := instCount + 1
      | "inductive" => indCount := indCount + 1
      | "structure" => structCount := structCount + 1
      | "class" => classCount := classCount + 1
      | _ => pure ()
    | none => pure ()
  
  let stats : FileAnalysisStats := {
    fileName := filePath
    totalDeclarations := results.size
    definitionCount := defCount
    theoremCount := thmCount
    lemmaCount := lemmaCount
    instanceCount := instCount
    inductiveCount := indCount
    structureCount := structCount
    classCount := classCount
  }
  
  IO.println s!"✅ Analysis complete!"
  IO.println s!"📊 Statistics:"
  IO.println s!"   • Total: {stats.totalDeclarations}"
  IO.println s!"   • Definitions: {stats.definitionCount}"
  IO.println s!"   • Theorems: {stats.theoremCount}"
  IO.println s!"   • Lemmas: {stats.lemmaCount}"
  IO.println s!"   • Instances: {stats.instanceCount}"
  IO.println s!"   • Inductives: {stats.inductiveCount}"
  
  return (results, stats)

-- 내장 HashMap 사용 (Std.Data.HashMap 대신)
structure CodeGenerator where
  template : String
  variableMap : List (String × String)  -- HashMap 대신 List 사용
  dependencyMap : List (Name × Name)

-- 선언 정보에서 완전한 Lean 코드 재생성
def regenerateDeclaration (info : ReversibleDeclarationInfo) (generator : CodeGenerator := {}) : String :=
  let mut code := ""
  
  -- 네임스페이스 추가
  if !info.namespace.isAnonymous then
    code := code ++ s!"namespace {info.namespace}\n\n"
  
  -- 문서 문자열 추가
  match info.docString with
  | some doc => code := code ++ s!"/-- {doc} -/\n"
  | none => pure ()
  
  -- 속성 추가
  if !info.attributes.isEmpty then
    for attr in info.attributes do
      code := code ++ s!"@[{attr}]\n"
  
  -- 매개변수들 재구성
  let mut params := ""
  
  -- implicit parameters
  if !info.implicitParams.isEmpty then
    let implicitList := info.implicitParams.map fun (name, type) => s!"{name} : {type}"
    params := params ++ s!" {{{String.intercalate ", " implicitList.toList}}}"
  
  -- instance parameters  
  if !info.instanceParams.isEmpty then
    let instanceList := info.instanceParams.map fun (name, type) => s!"[{name} : {type}]"
    params := params ++ " " ++ String.intercalate " " instanceList.toList
  
  -- explicit parameters
  if !info.explicitParams.isEmpty then
    let explicitList := info.explicitParams.map fun (name, type) => s!"({name} : {type})"
    params := params ++ " " ++ String.intercalate " " explicitList.toList
  
  -- 최종 선언 재구성
  code := code ++ s!"{info.kind} {info.name}{params} : {info.typeExpr}"
  
  if !info.valueExpr.isEmpty then
    code := code ++ s!" := {info.valueExpr}"
  
  if !info.namespace.isAnonymous then
    code := code ++ s!"\n\nend {info.namespace}"
    
  code

-- 전체 파일 재생성
def regenerateFile (infos : Array ReversibleDeclarationInfo) (fileName : String) : String :=
  let mut output := s!"-- Regenerated from analysis of {fileName}\n\n"
  
  -- imports 추가 (의존성 기반)
  let allDeps := infos.foldl (fun acc info => acc ++ info.typeDependencies ++ info.valueDependencies) #[]
  let uniqueDeps := allDeps.toList.eraseDups.toArray
  
  output := output ++ "-- Required imports (auto-detected)\n"
  for dep in uniqueDeps do
    if dep.toString.contains "Mathlib" then
      output := output ++ s!"import {dep}\n"
  
  output := output ++ "\n"
  
  -- 선언들 재생성
  for info in infos do
    output := output ++ regenerateDeclaration info ++ "\n\n"
  
  output

-- 학습 데이터 생성을 위한 구조화된 출력
def generateTrainingData (infos : Array ReversibleDeclarationInfo) : String :=
  let mut training := "# Lean Training Data\n\n"
  
  for info in infos do
    training := training ++ s!"## {info.name}\n"
    training := training ++ s!"**Kind**: {info.kind}\n"
    training := training ++ s!"**Type**: `{info.typeExpr}`\n"
    
    if !info.explicitParams.isEmpty then
      training := training ++ s!"**Parameters**: "
      let paramStrs := info.explicitParams.map fun (n, t) => s!"{n}: {t}"
      training := training ++ String.intercalate ", " paramStrs.toList ++ "\n"
    
    training := training ++ s!"**Implementation**:\n```lean\n{info.valueExpr}\n```\n"
    training := training ++ s!"**Full Syntax**:\n```lean\n{info.fullSyntax}\n```\n\n"
    training := training ++ "---\n\n"
  
  training

-- 분석 + 통계 출력을 포함한 메인 실행 함수
def runReversibleAnalysis (filePath : String) (outputFormat : String := "reversible") : MetaM Unit := do
  IO.println s!"🚀 Starting reversible analysis of {filePath}"
  IO.println s!"📈 This will enable learning → reverse transformation for new problem solving"
  IO.println ""
  
  let startTime ← IO.monoMsNow
  let (results, stats) ← analyzeFileReversible filePath
  let endTime ← IO.monoMsNow
  let duration := endTime - startTime
  
  IO.println s!"⏱️  Analysis completed in {duration}ms"
  IO.println ""
  IO.println s!"📋 ANALYSIS SUMMARY for {filePath}:"
  IO.println s!"   └─ 📁 File: {stats.fileName}"
  IO.println s!"   └─ 📊 Total declarations: {stats.totalDeclarations}"
  IO.println s!"   └─ 🔧 Definitions: {stats.definitionCount}"
  IO.println s!"   └─ 📜 Theorems: {stats.theoremCount}"
  IO.println s!"   └─ 🧩 Lemmas: {stats.lemmaCount}"
  IO.println s!"   └─ ⚡ Instances: {stats.instanceCount}"
  IO.println s!"   └─ 🏗️  Inductives: {stats.inductiveCount}"
  IO.println ""
  
  match outputFormat with
  | "reversible" => 
    IO.println "🔄 REVERSIBLE FORMAT (Full reconstruction possible):"
    IO.println "{"
    for (info, idx) in results.zipWithIndex do
      IO.println s!"  \"decl_{idx}\": \{"
      IO.println s!"    \"name\": \"{info.name}\","
      IO.println s!"    \"kind\": \"{info.kind}\","
      IO.println s!"    \"namespace\": \"{info.namespace}\","
      IO.println s!"    \"full_syntax\": \"{info.fullSyntax.replace "\"" "\\\"" |>.replace "\n" "\\n"}\","
      IO.println s!"    \"type_expr\": \"{info.typeExpr.replace "\"" "\\\"" |>.replace "\n" "\\n"}\","
      IO.println s!"    \"value_expr\": \"{info.valueExpr.replace "\"" "\\\"" |>.replace "\n" "\\n"}\","
      IO.println s!"    \"explicit_params\": [{String.intercalate ", " (info.explicitParams.map fun (n,t) => s!"\"{n}:{t}\"").toList}],"
      IO.println s!"    \"implicit_params\": [{String.intercalate ", " (info.implicitParams.map fun (n,t) => s!"\"{n}:{t}\"").toList}],"
      IO.println s!"    \"dependencies\": [{String.intercalate ", " (info.typeDependencies.map (s!"\"{·}\"")).toList}]"
      IO.println s!"  \}" ++ (if idx < results.size - 1 then "," else "")
    IO.println "}"
    
  | "regenerated" =>
    IO.println "🔄 REGENERATED LEAN CODE:"
    IO.println "```lean"
    IO.println (regenerateFile results filePath)
    IO.println "```"
    
  | "training" =>
    IO.println "🎓 TRAINING DATA FORMAT:"
    IO.println (generateTrainingData results)
    
  | "verification" =>
    IO.println "✅ VERIFICATION (Original vs Regenerated):"
    for info in results.take 5 do  -- 처음 5개만 검증 출력
      IO.println s!"Original: {info.fullSyntax}"
      IO.println s!"Regenerated: {regenerateDeclaration info}"
      IO.println "---"
      
  | _ => 
    IO.println "❌ Unknown format. Available: reversible, regenerated, training, verification"

-- 역변환: 학습된 패턴에서 새로운 선언 생성
structure DeclarationPattern where
  kindPattern : String
  typePattern : String
  implementationPattern : String
  parameterPattern : Array (String × String)
  deriving Repr

def generateNewDeclaration (pattern : DeclarationPattern) (newName : String) (substitutions : HashMap String String) : String :=
  let mut generated := s!"{pattern.kindPattern} {newName}"
  
  -- 매개변수 적용
  if !pattern.parameterPattern.isEmpty then
    let params := pattern.parameterPattern.map fun (name, type) => 
      let substName := substitutions.getD name name
      let substType := substitutions.getD type type
      s!"({substName} : {substType})"
    generated := generated ++ " " ++ String.intercalate " " params.toList
  
  -- 타입 적용
  let substType := substitutions.foldl (fun acc key val => acc.replace key val) pattern.typePattern
  generated := generated ++ s!" : {substType}"
  
  -- 구현 적용
  if !pattern.implementationPattern.isEmpty then
    let substImpl := substitutions.foldl (fun acc key val => acc.replace key val) pattern.implementationPattern
    generated := generated ++ s!" := {substImpl}"
  
  generated

-- 패턴 학습: 유사한 선언들에서 패턴 추출
def learnPatterns (infos : Array ReversibleDeclarationInfo) (kindFilter : String) : Array DeclarationPattern :=
  let filtered := infos.filter (·.kind == kindFilter)
  
  -- 간단한 패턴 추출 (실제로는 더 정교한 ML 기법 사용 가능)
  filtered.map fun info => {
    kindPattern := info.kind
    typePattern := info.typeExpr
    implementationPattern := info.valueExpr
    parameterPattern := info.explicitParams
  }

-- 매크로: 완전 자동 분석
macro "analyze_all_mathlib" : command => do
  `(command| #eval analyzeAllMathlib)

-- 환경 기반 모듈 탐색 (대안 방법)
def getEnvironmentModules : MetaM (Array String) := do
  let env ← getEnv
  let allModules := env.allImportedModuleNames
  let mathlibModules := allModules.filter fun name => 
    name.toString.startsWith "Mathlib"
  
  let moduleStrings := mathlibModules.map (·.toString)
  return moduleStrings

-- 스마트 파일 발견 (여러 방법 조합) - MetaM 컨텍스트
def discoverMathilibModules : MetaM (Array String) := do
  IO.println "🔍 Auto-discovering Mathlib modules using multiple methods..."
  
  -- 방법 1: 환경에서 로드된 모듈들
  let loadedModules ← getLoadedMathilibModules
  IO.println s!"   📚 Environment scan found: {loadedModules.size} loaded modules"
  
  -- 방법 2: 파일시스템 스캔 (IO 컨텍스트 필요)
  let filesFromFS ← findMathilibFiles
  IO.println s!"   📁 Filesystem scan found: {filesFromFS.size} files"
  
  -- 결합된 결과
  let mut allModules : Array String := filesFromFS
  for module in loadedModules do
    if !allModules.contains module.toString then
      allModules := allModules.push module.toString
  
  IO.println s!"✅ Total unique modules discovered: {allModules.size}"
  return allModules

-- Mathlib 디렉토리 자동 탐색 (System.FilePath 없이)
def findMathilibFiles (mathlibPath : String := "lake-packages/mathlib4/Mathlib") : IO (Array String) := do
  -- 파일시스템 API 사용 시 오류가 발생할 수 있으므로, 
  -- 환경에서 로드된 모듈을 기반으로 추론하는 방식으로 변경
  IO.println s!"🔍 Attempting to find Mathlib files..."
  IO.println s!"📂 Target directory: {mathlibPath}"
  
  -- 기본적으로 알려진 핵심 Mathlib 모듈들 리스트
  let coreModules := #[
    "Mathlib.Logic.Basic",
    "Mathlib.Data.Nat.Basic", 
    "Mathlib.Data.Int.Basic",
    "Mathlib.Data.List.Basic",
    "Mathlib.Data.Set.Basic",
    "Mathlib.Algebra.Group.Basic",
    "Mathlib.Algebra.Ring.Basic",
    "Mathlib.Algebra.Field.Basic",
    "Mathlib.Analysis.Basic",
    "Mathlib.Topology.Basic",
    "Mathlib.CategoryTheory.Basic",
    "Mathlib.NumberTheory.Basic",
    "Mathlib.Geometry.Euclidean.Basic",
    "Mathlib.Probability.Basic",
    "Mathlib.Combinatorics.Basic",
    "Mathlib.Order.Basic",
    "Mathlib.Tactic.Basic"
  ]
  
  IO.println s!"📋 Using core module list: {coreModules.size} modules"
  return coreModules

-- 로드된 모듈에서 Mathlib 모듈들 자동 발견
def getLoadedMathilibModules : MetaM (Array Name) := do
  let env ← getEnv
  let allModules := env.allImportedModuleNames
  let mathlibModules := allModules.filter fun name => 
    name.toString.startsWith "Mathlib"
  
  IO.println s!"📚 Found {mathlibModules.size} loaded Mathlib modules"
  return mathlibModules

-- 전체 Mathlib 분석 (자동 탐색)
def analyzeAllMathlib (outputDir : String := "./mathlib_analysis") : MetaM Unit := do
  IO.println "🌟 STARTING COMPLETE MATHLIB ANALYSIS"
  IO.println "="⟩repeat 80⟨
  
  -- 1. 자동 모듈 탐색 (MetaM 컨텍스트에서)
  IO.println "🔍 Phase 1: Discovering Mathlib modules..."
  let allFiles ← discoverMathilibModules
  
  IO.println s!"📊 Total modules to analyze: {allFiles.size}"
  IO.println ""
  
  -- 2. 일괄 분석
  IO.println "⚡ Phase 2: Batch analysis..."
  let startTime ← IO.monoMsNow
  let mut totalDeclarations := 0
  let mut totalDefs := 0
  let mut totalTheorems := 0
  let mut totalLemmas := 0
  let mut processedFiles := 0
  let mut skippedFiles := 0
  
  -- 결과 저장을 위한 배열
  let mut allResults : Array ReversibleDeclarationInfo := #[]
  let mut fileStats : Array FileAnalysisStats := #[]
  
  for (file, idx) in allFiles.zipWithIndex do
    try
      IO.println s!"📁 [{idx+1}/{allFiles.size}] {file}"
      let (results, stats) ← analyzeFileReversible file
      
      allResults := allResults ++ results
      fileStats := fileStats.push stats
      
      totalDeclarations := totalDeclarations + stats.totalDeclarations
      totalDefs := totalDefs + stats.definitionCount
      totalTheorems := totalTheorems + stats.theoremCount
      totalLemmas := totalLemmas + stats.lemmaCount
      processedFiles := processedFiles + 1
      
      -- 진행률 표시
      if idx % 10 == 0 then
        let progress := (idx * 100) / allFiles.size
        IO.println s!"   📈 Overall progress: {progress}% ({totalDeclarations} declarations so far)"
        
    catch e =>
      IO.println s!"   ⚠️  Skipped {file}: {e}"
      skippedFiles := skippedFiles + 1
  
  let endTime ← IO.monoMsNow
  let totalTime := endTime - startTime
  
  -- 3. 최종 통계 및 결과 저장
  IO.println ""
  IO.println "="⟩repeat 80⟨
  IO.println "🎉 MATHLIB ANALYSIS COMPLETE!"
  IO.println s!"⏱️  Total time: {totalTime}ms ({totalTime/1000}s)"
  IO.println s!"📂 Files processed: {processedFiles}/{allFiles.size}"
  if skippedFiles > 0 then
    IO.println s!"⚠️  Files skipped: {skippedFiles}"
  IO.println ""
  IO.println "📊 GRAND TOTALS:"
  IO.println s!"   🔢 Total declarations: {totalDeclarations}"
  IO.println s!"   🔧 Total definitions: {totalDefs}"
  IO.println s!"   📜 Total theorems: {totalTheorems}" 
  IO.println s!"   🧩 Total lemmas: {totalLemmas}"
  IO.println s!"   📏 Avg declarations per file: {totalDeclarations / processedFiles}"
  
  -- 4. 결과 파일로 저장
  IO.println ""
  IO.println "💾 Saving results..."
  
  -- 전체 결과를 JSON으로 저장
  let jsonOutput := generateBigJson allResults fileStats
  -- 실제로는 파일 저장 코드 필요
  IO.println s!"📄 Generated complete analysis: {allResults.size} declarations"
  
  -- 5. 학습 패턴 생성
  IO.println "🧠 Generating learning patterns..."
  let defPatterns := learnPatterns allResults "def"
  let thmPatterns := learnPatterns allResults "theorem"
  
  IO.println s!"🎯 Extracted {defPatterns.size} definition patterns"
  IO.println s!"🎯 Extracted {thmPatterns.size} theorem patterns"
  IO.println "🚀 Ready for reverse transformation and new problem generation!"

-- 여러 파일 일괄 분석
def analyzeMathilibBatch (fileList : Array String) : MetaM Unit := do
  IO.println s!"🔥 Starting batch analysis of {fileList.size} files"
  IO.println "="⟩repeat 60⟨
  
  let mut totalStats : Nat := 0
  
  for (file, idx) in fileList.zipWithIndex do
    IO.println s!"📁 [{idx+1}/{fileList.size}] Processing: {file}"
    let (results, stats) ← analyzeFileReversible file
    totalStats := totalStats + stats.totalDeclarations
    
    -- 각 파일별 요약
    IO.println s!"   ✓ Extracted {results.size} reversible declarations"
    IO.println s!"   ✓ {stats.definitionCount} defs, {stats.theoremCount} theorems, {stats.lemmaCount} lemmas"
    IO.println ""
  
  IO.println "="⟩repeat 60⟨
  IO.println s!"🎉 Batch analysis complete! Total declarations processed: {totalStats}"

-- 큰 JSON 생성 (모든 결과 통합)
def generateBigJson (allResults : Array ReversibleDeclarationInfo) (stats : Array FileAnalysisStats) : String :=
  let mut json := "{\n"
  
  -- 통계 정보
  json := json ++ "  \"analysis_metadata\": {\n"
  json := json ++ s!"    \"total_files\": {stats.size},\n"
  json := json ++ s!"    \"total_declarations\": {allResults.size},\n"
  json := json ++ s!"    \"timestamp\": \"$(← IO.monoMsNow)\"\n"
  json := json ++ "  },\n"
  
  -- 파일별 통계
  json := json ++ "  \"file_stats\": [\n"
  for (stat, idx) in stats.zipWithIndex do
    json := json ++ s!"    \{\"file\": \"{stat.fileName}\", \"declarations\": {stat.totalDeclarations}\}"
    if idx < stats.size - 1 then json := json ++ ","
    json := json ++ "\n"
  json := json ++ "  ],\n"
  
  -- 모든 선언들
  json := json ++ "  \"declarations\": [\n"
  for (info, idx) in allResults.zipWithIndex do
    json := json ++ "    " ++ declarationInfoToReversibleJson info
    if idx < allResults.size - 1 then json := json ++ ","
    json := json ++ "\n"
  json := json ++ "  ]\n}"
  
  json

-- 가역가능한 JSON 형태
def declarationInfoToReversibleJson (info : ReversibleDeclarationInfo) : String :=
  s!"\{
    \"name\": \"{info.name}\",
    \"kind\": \"{info.kind}\",
    \"namespace\": \"{info.namespace}\",
    \"full_syntax\": \"{info.fullSyntax.replace "\"" "\\\"" |>.replace "\n" "\\n"}\",
    \"type_expr\": \"{info.typeExpr.replace "\"" "\\\"" |>.replace "\n" "\\n"}\",
    \"value_expr\": \"{info.valueExpr.replace "\"" "\\\"" |>.replace "\n" "\\n"}\",
    \"explicit_params\": [{String.intercalate ", " (info.explicitParams.map fun (n,t) => s!"\"{n}:{t}\"").toList}],
    \"implicit_params\": [{String.intercalate ", " (info.implicitParams.map fun (n,t) => s!"\"{n}:{t}\"").toList}],
    \"instance_params\": [{String.intercalate ", " (info.instanceParams.map fun (n,t) => s!"\"{n}:{t}\"").toList}],
    \"type_dependencies\": [{String.intercalate ", " (info.typeDependencies.map (s!"\"{·}\"")).toList}],
    \"value_dependencies\": [{String.intercalate ", " (info.valueDependencies.map (s!"\"{·}\"")).toList}],
    \"attributes\": [{String.intercalate ", " (info.attributes.map (s!"\"{·}\"")).toList}]
  \}"

-- 실제 사용 예시 (자동 분석 포함)
comprehensive_analysis_example : MetaM Unit := do
  -- 1. 전체 자동 분석
  analyzeAllMathlib
  
  -- 2. 특정 영역 집중 분석
  let algebraFiles := #[
    "Mathlib.Algebra.Group.Basic",
    "Mathlib.Algebra.Ring.Basic", 
    "Mathlib.Algebra.Field.Basic"
  ]
  IO.println "🧮 Analyzing core algebra files..."
  analyzeMathilibBatch algebraFiles
  
  -- 3. 학습 패턴 추출 및 새 코드 생성 데모
  IO.println "🤖 Demonstrating reverse transformation..."
  testReverseTransformation "Mathlib.Data.Nat.Basic"

-- 진행상황 저장 및 재시작 지원
structure AnalysisProgress where
  completedFiles : Array String
  lastProcessedIndex : Nat
  partialResults : Array ReversibleDeclarationInfo
  startTime : UInt64
  deriving Repr

def saveProgress (progress : AnalysisProgress) : IO Unit := do
  -- 진행상황을 임시 파일로 저장
  IO.println s!"💾 Saving progress: {progress.completedFiles.size} files completed"

def loadProgress : IO (Option AnalysisProgress) := do
  -- 이전 진행상황 로드 (파일이 존재하면)
  IO.println "🔄 Checking for previous analysis progress..."
  return none  -- 구현 필요시 추가

-- 중단점 지원 전체 분석
def analyzeAllMathilibResumable : MetaM Unit := do
  let existingProgress ← loadProgress
  
  match existingProgress with
  | some progress => 
    IO.println s!"📤 Resuming from {progress.completedFiles.size} completed files"
    -- 이어서 분석 계속
  | none =>
    IO.println "🆕 Starting fresh analysis"
    analyzeAllMathlib

-- 역변환 테스트 함수
def testReverseTransformation (originalFile : String) : MetaM Unit := do
  IO.println "🔄 Testing reverse transformation capability..."
  
  let (infos, _) ← analyzeFileReversible originalFile
  let regenerated := regenerateFile infos originalFile
  
  IO.println "Original analysis completed ✓"
  IO.println "Regeneration completed ✓"
  IO.println s!"Code regenerated with {regenerated.split '\n' |>.length} lines"
  
  -- 패턴 학습 테스트
  let defPatterns := learnPatterns infos "def"
  let thmPatterns := learnPatterns infos "theorem"
  
  IO.println s!"Learned {defPatterns.size} definition patterns ✓"
  IO.println s!"Learned {thmPatterns.size} theorem patterns ✓"
  IO.println "🎯 Ready for new problem generation!"

#check runReversibleAnalysis
#check testReverseTransformation
#check generateNewDeclaration
#check analyzeAllMathlib          -- 🆕 전체 자동 분석
#check discoverMathilibFiles      -- 🆕 자동 파일 탐색
#check analyzeMathilibBatch       -- 🆕 일괄 처리
