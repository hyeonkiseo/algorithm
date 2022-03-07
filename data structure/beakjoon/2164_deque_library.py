# by python library
from collections import deque

a = int(input())

que = deque()
for i in range(a):
    que.append(i + 1)

while len(que) > 1:
    que.popleft()
    que.append(que.popleft())
    
print(que[0])