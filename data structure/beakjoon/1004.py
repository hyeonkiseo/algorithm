import math
t = int(input())
for _ in range(t):
    x1, y1, x2, y2 = map(int,input().split())
    cnt = 0
    n = int(input())
    for i in range(n):
        cx,cy,r = map(int,input().split())
        d1 = math.sqrt((cx-x1)**2 + (cy-y1)**2)
        d2 = math.sqrt((cx-x2)**2 + (cy-y2)**2)
        if ((d1 < r) and (d2 > r)) or ((d1 > r) and (d2 < r)):
            cnt+=1
    print(cnt)