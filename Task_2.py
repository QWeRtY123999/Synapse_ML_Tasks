import pandas as pd
import matplotlib.pyplot as plt

# Task 2.1
df = pd.read_csv("netflix_titles.csv")
print(df.head(10))
df = df.dropna()
df.reset_index()
print(df)

# Task 2.2
print(df.query('release_year == 2021 and type == \'Movie\''))

print(df[df['title'].str.contains('Avengers') & (df['type'] == 'Movie')])


"""
freq = df['country'].value_counts()

print(freq)

freq = df['country'].value_counts()
top = freq.head(5).index.tolist()
filt = df[df['country'].isin(top)]

print("DataFrame with rows from top 5 countries by frequency:")
print(filt)
"""


a = list(df.country)
b = []
for i in a:
    c = i.split(', ')
    for j in c:
        b.append(j)
#print(b)
c = set(b)
c.remove('')
d = []
e = []
for i in c:
    print(f"{i} : {b.count(i)}")
    d.append(i)
    e.append(b.count(i))

f = sorted(list(zip(d,e)),key=lambda data : data[1],reverse=True)
#print(f)

num = [f[i][1] for i in range(5)]
country = [f[i][0] for i in range(5)]

#print(country,num)
print(df.set_index('country').filter(regex="United States|India|United Kingdom|Canada|France", axis=0))
df.reset_index()

# Task 2.3
fig = plt.figure(figsize=(10,7))
plt.pie(num,labels=country)

plt.show()
#print(list(df.keys()))