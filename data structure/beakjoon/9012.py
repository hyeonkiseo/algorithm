class Node:
    def __init__(self, data, nxt):
        self.data = data
        self.nxt = nxt
        
class Mystack:
    def __init__(self):
        self.size = 0
        self.head = Node(None,None)  # it located after newest node
        
    def push(self,data):
        new = Node(data,self.head.nxt) # 기존의 가장 앞에 있었던 노드를 nxt로 받는 노드 생성
        self.head.nxt = new # 헤드의 다음 노드를 new node로 바꿈
        self.size +=1
        pass
        
        
    def pop(self):
        if self.head.nxt == None:
            return None
        
        pop = self.head.nxt # 현재 가장 최신에 들어온 데이터
        self.head.nxt = pop.nxt # 바로 전에 들어온 데이터로 head의 포인터 지정 
        self.size -= 1 # size 1 감소
        return pop.data  # pop의 데이터 반환
    
    def peek(self):
        return self.head.nxt.data  # 구조 변환 없이 데이터만 반환
    
    def isEmpty(self):
        if self.head.nxt == None:
            return True
        else : 
            return False
        
    def get_size(self):
        return self.size        
        

def VPS(data):
    stack = Mystack()
    for i in data:
        if i == '(':
            stack.push(1)
        else:
            if stack.size == 0:
                print('NO')
                return None
            temp = stack.pop()

    if stack.size == 0:
        print("YES")
        return None
    else:
        print("NO")
        return None
            
        
t = int(input())

for _ in range(t):
    data = input()
    VPS(data)