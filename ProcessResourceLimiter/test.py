import random

a = 0
for i in range(random.randint(0, 30000000)):
    a += random.random()

print(a)
