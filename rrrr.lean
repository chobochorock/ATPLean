import Init.System.FilePath

def main (args : List String) : IO UInt32 := do
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
