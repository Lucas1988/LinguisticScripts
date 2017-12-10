# python3 TestResultsAnalysis.py /Users/lucasvergeest/Documents/nl-NL.table > test_results_3.txt

# With this script you can interpret your test results more easily. It will display the averaged rating of each audio fragment per voice.
# You just have to change the system argument, and maybe do some minor changes below, depending on the exact format of your test results.

import sys
import operator
import itertools
import numpy as np

results = []
means = []

with open(sys.argv[1]) as lines:
    for line in lines:
        elements = line.split()
        results.append([elements[1][-7:-4], int(elements[0]), elements[2]])
        

sorted_results = sorted(results, key=operator.itemgetter(0))
grouped_results = itertools.groupby(sorted_results, key=operator.itemgetter(0,2))
grouped_results = [[k, [x[1] for x in v]] for k, v in grouped_results]

for group in grouped_results:
    means.append([group[0], np.mean(group[1])])

sorted_means = sorted(means, key=operator.itemgetter(1))

for mean in sorted_means:
    print(mean[0], round(mean[1], 2))
