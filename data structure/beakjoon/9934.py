from collections import deque
class Node : 
    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right

def buildtree(lst, start, end):
    if start > end:
        return None
    med = int((start + end) /2)
    
    new = Node(lst[med], None, None)
    new.left = buildtree(lst,start,med-1)
    new.right = buildtree(lst,med+1,end)
    
    return new
    
k = int(input())
n = 2**k -1

q = deque()
q.append(buildtree(list(map(int,input().split())), 0, n-1))

while len(q) >0:
    size = len(q)
    for _ in range(size):
        pops = q.popleft()
        if pops.left != None:
            q.append(pops.left)
            q.append(pops.right)
        print(pops.data, end = ' ')
    print()