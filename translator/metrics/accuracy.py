val = open('val.txt', encoding='UTF-8')
out = open('out.txt', encoding='UTF-8')

t = val.readline()
a = out.readline()
sum = 0
right = 0

while t and a:
    error = []
    t = t.strip()
    a = a.strip()
    for i in range(len(t)):
        sum += 1
        if t[i] == a[i]:
            right += 1
        else:
            error.append(a[i])
    print(t)
    print(a)
    print("wrong with {}", format(error))

    t = val.readline()
    a = out.readline()
print("right={}".format(right))
print("sum={}".format(sum))
print(right / sum)

