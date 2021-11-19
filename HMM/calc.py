import math

f = open("bigram_probs.txt", "r")

sum = 0
for row in f.readlines():
    row = row.split(" ")
    if row[1] == "0":

        #print(float(row[2]))
        sum += float(row[2])

p = math.exp((sum))
print(p)
print(sum)