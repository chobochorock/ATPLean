import Init.System.FilePath

-- 지정된 경로에서 시작하여 모든 .lean 파일을 재귀적으로 찾는 함수
partial def findLeanFiles (path : System.FilePath) : IO (Array System.FilePath) := do
  let mut leanFiles := #[]
  -- 경로가 실제로 존재하는지 확인
  if !(← path.pathExists) then
    IO.eprintln s!"경고: '{path}' 경로를 찾을 수 없습니다."
    return leanFiles

  -- 경로가 디렉토리인지 확인
  if ← path.isDir then
    -- 디렉토리 내의 모든 항목을 순회
    for entry in (← path.readDir) do
      -- 각 항목에 대해 재귀적으로 함수 호출하여 결과를 병합
      leanFiles := leanFiles ++ (← findLeanFiles entry.path)
  -- 경로가 파일이고, 확장자가 .lean인 경우
  else if path.extension == some "lean" then
    leanFiles := leanFiles.push path

  return leanFiles

-- 실행 파일의 시작점
def QueryingFiles (args : List String) : IO UInt32 := do
  -- 1. 커맨드 라인 인자가 있는지 확인
  if args.isEmpty then
    IO.println "오류: 검색을 시작할 디렉토리 경로를 입력해주세요."
    IO.println "사용법: ./build/bin/analyzer path/to/your/lean/project"
    return 1 -- 오류 코드

  let rootPath : System.FilePath := args.head!
  IO.println s!"📂 '{rootPath}' 디렉토리에서 .lean 파일을 검색합니다..."

  -- 2. 재귀적으로 .lean 파일 찾기
  let foundFiles ← findLeanFiles rootPath

  -- 3. 결과 출력
  if foundFiles.isEmpty then
    IO.println "- .lean 파일을 찾지 못했습니다."
  else
    IO.println s!"\n✅ 총 {foundFiles.size}개의 .lean 파일을 찾았습니다:"
    for file in foundFiles do
      IO.println s!"  - {file}"

  return 0 -- 성공 코드

def ReadingFile (args : List String) : IO UInt32 := do
  -- 1. 인자가 있는지 확인
  if args.isEmpty then
    IO.eprintln "오류: 파일 경로를 입력해주세요."
    IO.eprintln "사용법: ./build/bin/my-app path/to/file.lean"
    return 1 -- 실패 코드(1)를 반환하며 종료

  -- 2. 첫 번째 인자를 파일 경로로 사용
  let filePath : System.FilePath := args.head!

  -- 3. 파일이 실제로 존재하는지 확인
  if !(← filePath.pathExists) then
    IO.eprintln s!"오류: '{filePath}' 파일을 찾을 수 없습니다."
    return 1

  -- 4. 파일 내용 읽기
  IO.println s!"--- [ {filePath} ] 파일 내용 시작 ---"
  try
    let content ← IO.FS.readFile filePath
    -- 5. 내용 출력
    IO.print content
  catch e =>
    IO.eprintln s!"파일을 읽는 중 오류 발생: {e}"
    return (1 : UInt32)

  IO.println s!"\n--- [ {filePath} ] 파일 내용 끝 ---"
  return 0 -- 성공 코드(0)를 반환하며 종료

def main (args : List String) : IO UInt32 := do
  return (0 : UInt32)
