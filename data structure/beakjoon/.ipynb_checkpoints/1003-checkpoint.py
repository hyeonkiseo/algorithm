zero = [1,0,1]
one = [0,1,1]

def fibonacci(num):
    if num >= len(zero):
        for i in range(len(zero) , num +1):
            zero.append(zero[i-1] + zero[i-2])
            one.append(one[i-1] + one[i-2])
    print(f'{zero[num]} {one[num]}')
    pass

n = int(input())
for _ in range(n):
    fibonacci(int(input()))