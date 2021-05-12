import sys

source_filename = sys.argv[1]
destination_filename = sys.argv[2]
with open(source_filename) as f:
    lines = f.readlines()[1:]
data = [tuple(float(x) for x in line.strip().split()) for line in lines]
averages = []
curr_sum = (0, 0, 0)
for index, d in enumerate(data):
    curr_sum = tuple(x + y for x, y in zip(curr_sum, d))
    curr_avg = tuple(x / (index + 1) for x in curr_sum)
    averages.append(curr_avg)
with open(destination_filename, 'w') as f:
    for a in averages:
        print(*a, sep='\t', file=f)