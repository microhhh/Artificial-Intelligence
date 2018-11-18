

val = open('val.txt', encoding='UTF-8')
out = open('out.txt', encoding='UTF-8')

t=val.read()
a=out.read()
sum=0
right=0
for i in range(len(t)):
    sum+=1
    if t[i]=='\n':
        continue
    if t[i] == a[i]:
        print('t',t[i])
        right+=1

print(right/sum)
        # for i in range(len())
        # answer = answer.strip()
        # line = line.strip()

