jumbled_superheroes = ['DocToR_sTRAngE', 'sPidERmaN', 'MoON_KnigHT', 'caPTAIN_aMERIca', 'hULK']

indices = []
decoded_names = []
for e in enumerate(jumbled_superheroes):
    decoded_names.append(str.lower(e[1]).replace('_'," "))
    indices.append(e[0])
    #print(f"{e[0]}. {decoded_names[e[0]]}")

func = lambda input: len(input)
name_lengths = list(map(func,decoded_names))
#print(name_lengths)

temp = list(zip(indices,name_lengths))
#print(temp)

func2 = lambda data:data[1]
sorted_names = sorted(temp,key=func2)
#print(sorted_names)

c = 1
for i in sorted_names:
    print(f"{c}. {str.title(decoded_names[i[0]])}")
    c += 1