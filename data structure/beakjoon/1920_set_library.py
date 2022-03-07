t = int(input())
data = list(map(int,input().split()))
S = set()

for i in data:
    S.add(i)
    
m = int(input())
test = list(map(int,input().split()))
for i in test:
    if i in S:
        print(1)
    else:
        print(0)