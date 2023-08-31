encoded_lists = [[1,2,3,4,6],
                 [5,7,8,9,15],
                 [12,14,16,18],
                 [10,11,12,13,16,17,18,20]]

def explode_chains(list):
    for i in list:
        f=0
        j = 0
        for _ in range(len(i)-1):
            if((i[j+1] - i[j]) == 1):
                f += 1
                j += 1
            else:
                f = 0
                j += 1
            if f == 2:
                i.remove(i[j])
                i.remove(i[j-1])
                i.remove(i[j-2])
                j -= 3

            #print(j,end="")
        print(i)

explode_chains(encoded_lists)