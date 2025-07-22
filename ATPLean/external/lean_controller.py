from lean_interact import *
# import Main # this is lean code

config = LeanREPLConfig(verbose=True)
server = LeanServer(config)
response1 = server.run(Command(cmd="theorem ex (n : Nat) : n = 5 → n = 5 := sorry "))
response2 = server.run(Command(cmd="example (x : Nat) : x = 5 → x = 5 := by exact ex x", env=0))


print(response1.messages)
print(response2.messages)
# f = open('lean_result.txt', mode='w')
# f.write(response)
# f.close()
