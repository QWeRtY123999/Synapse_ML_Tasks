from itertools import combinations

lst = ['0001', '0011', '0101', '1011', '1101', '1111']
new_lst = [int(i, 2) for i in lst]

a = []
for r in range(1,len(new_lst)):
    for c in combinations(new_lst,r):
        a.append(c)

b = []
c = []
for items in a:
    res = [i for i in new_lst if i not in items]
    b.append(sum(items))
    c.append(sum(res))

d = list(zip(b,c))
new_lst = sorted(d,key= lambda data: abs(data[0]-data[1]))

print(f"The final list is: {new_lst[0]}")