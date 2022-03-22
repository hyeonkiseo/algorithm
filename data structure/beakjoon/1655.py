import sys
import heapq

input = sys.stdin.readline

n = int(input())

leftheap = []  # 작은 숫자들
rightheap = []  # 큰 숫자들
for i in range(n):
    num = int(input())
    if len(leftheap) == len(rightheap):
        heapq.heappush(leftheap, (-num,num))
    else:
        heapq.heappush(rightheap, num)

    if len(rightheap) >= 1 and len(leftheap) >= 1 and leftheap[0][1] > rightheap[0]:
        rightpop = heapq.heappop(rightheap)
        leftpop = heapq.heappop(leftheap)
        heapq.heappush(rightheap, leftpop[1])
        heapq.heappush(leftheap, (-rightpop, rightpop))

    print(leftheap[0][1])