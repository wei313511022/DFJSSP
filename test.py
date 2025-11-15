import time

N = 29160  # number of iterations

total = 0

start = time.perf_counter()  # high-resolution timer

for i in range(N):
    # some simple operations
    total += 2
    total -= 2
    total *= 1  # does nothing, just an example

end = time.perf_counter()

print("Final total:", total)
print("Total running time: {:.3f} seconds".format(end - start))
