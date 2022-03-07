class DoubleLinkNode:
    def __init__(self, data, nxt, prev):
        self.data = data
        self.nxt = nxt
        self.prev = prev    

class Myqueue:
    def __init__(self):
        self.size = 0
        self.head = DoubleLinkNode(None,None,None)
        self.tail = DoubleLinkNode(None,None,None)
        self.head.prev = self.tail
        self.tail.nxt = self.head
        
    def enqueue(self,data):
        new = DoubleLinkNode(data,self.tail.nxt,self.tail)
        self.tail.nxt.prev = new
        self.tail.nxt = new
        self.size +=1
        return None
    
    def dequeue(self):
        pop = self.head.prev
        pop.prev.nxt = self.head
        self.head.prev = pop.prev
        self.size -=1
        return pop.data
    
    def peek(self):
        return self.head.prev.data
    
    def isEmpty(self):
        if self.head.prev == self.tail:
            return True
        else:
            return False
        
    def get_size(self):
        return self.size    

a = int(input())

que = Myqueue()
for i in range(a):
    que.enqueue(i + 1)

while que.get_size() > 1:
    que.dequeue()
    que.enqueue(que.dequeue())
print(que.peek())
