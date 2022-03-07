class Node ():
    def __init__(self, data, nxt):
        self.data = data
        self.nxt = nxt
        
class MyLinkedList():
    def __init__(self):
        self.size = 0
        self.head = Node(None,None)
        pass
    
    def add(self,data):
        new = Node(data, None)
        curr = self.head
        
        while curr.nxt != None :
            curr = curr.nxt
            
        curr.nxt = new
                
        self.size += 1        
        

def findit(value, t = 100000):
    curr = array[hash(i) % t]
    temp = curr.head.nxt
    while temp != None:
        if temp.data == value:
            print(1)
            return None
        temp = temp.nxt
    print(0)
    return None
        
    

t = int(input())
array = []
for _ in range(100000):
    new = MyLinkedList()
    array.append(new)

data = list(map(int,input().split()))

for i in data:
    curr = array[hash(i) % 100000]
    curr.add(i)
    
m = int(input())

test = list(map(int,input().split()))
for i in test:
    findit(i)