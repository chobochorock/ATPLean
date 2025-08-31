import json
import subprocess
import threading


def run_lean_server():
    # Lean 서버 실행 (lake 프로젝트 환경에서 실행해야 함)
    return subprocess.Popen(
        ["lake", "exe", "lean", "--server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def reader_thread(pipe):
    """Lean 서버 출력 읽기 (비동기)"""
    while True:
        line = pipe.readline()
        if not line:
            break
        print("LEAN >", line.strip())


def send_request(proc, req):
    """JSON-RPC 메시지를 Lean에 전송"""
    msg = json.dumps(req)
    header = f"Content-Length: {len(msg)}\r\n\r\n"
    proc.stdin.write(header + msg)
    proc.stdin.flush()


# Lean 서버 시작
proc = run_lean_server()
threading.Thread(target=reader_thread, args=(proc.stdout,), daemon=True).start()

# Lean 서버에 열려 있는 파일을 등록 (예시: Test.lean)
req_initialize = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"rootUri": None, "capabilities": {}},
}
send_request(proc, req_initialize)

req_open_file = {
    "jsonrpc": "2.0",
    "method": "textDocument/didOpen",
    "params": {
        "textDocument": {
            "uri": "file:///./Test.lean",
            "languageId": "lean",
            "version": 1,
            "text": "example (n : Nat) : n = n := by rfl",
        }
    },
}
send_request(proc, req_open_file)

# goal 요청
req_goal = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "textDocument/hover",
    "params": {
        "textDocument": {"uri": "file:///./Test.lean"},
        "position": {"line": 0, "character": 30},
    },
}
send_request(proc, req_goal)
