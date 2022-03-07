res = []
cond = []

def minimum_time(num):
    d = len(res)
    if res[num-1] == 0:  # value is not saved
        if len(cond[num-1]) == 0:   # have no condition
            res[num-1] = time[num-1]
            return res[num-1]
        else:                       # have condition
            temp = []
            for i in cond[num-1]:
                temp.append(minimum_time(i))
            res[num-1] = max(temp) + time[num-1]
            return res[num-1]
    else:
        return res[num-1]

t = int(input())

for _ in range(t):
    numb, numr = map(int,input().split())
    time = list(map(int,input().split()))
    
    cond = []
    for i in range(numb):
        cond.append([])
    for i in range(numr):
        pre, aft = map(int,input().split())
        cond[aft-1].append(pre)
    obj = int(input())

    # dinamic programming
    res = []
    for i in range(numb):
        res.append(0)
    print(minimum_time(obj))