'''
python3 Outliers.py output.txt.phones >> result.txt
'''

import sys
import re
import numpy
import scipy
import operator
from itertools import groupby
import matplotlib.pyplot as plt
import collections

result = []
consonantlist = []
longvowellist = []
shortvowellist = []

#Read in all the phonemes and calculate their durations
with open(sys.argv[1]) as lines:
    for line in lines:
        values = line.split()
        number = (float(values[1])-float(values[0]))
        if values[2] != 'sil\n':
            if re.search('^(aa|ee|oo|uu|oe|ui|ij|eu)',values[2]):
                result.append([round(number,6),values[0],values[1],values[2],'longvowel'])
            if re.search('^(ax|ex|ox|ux|ix|@@)',values[2]):
                result.append([round(number,6),values[0],values[1],values[2],'shortvowel'])
            if re.search('^[bcdfghjklmnpqrstvwxz]$',values[2]):
                result.append([round(number,6),values[0],values[1],values[2],'consonant'])

#Sort all phonemes by their duration
listsorted = sorted(result, key = operator.itemgetter(3, 0))

values = set(map(lambda x:x[3], listsorted))
newlist = [[y for y in listsorted if y[3]==x] for x in values]

newlist2=[]
averages=[]

#Plot bars
for i in newlist:
    for a in i:
        newlist2.append(a[0])
    counter = collections.Counter(newlist2)
    
    #averages.append([i[0][3], numpy.mean(newlist2)])
    #newlist2=[]
    
    plt.bar([a for a in counter.keys()], [b for b in counter.values()], width=0.005)
    plt.xlabel(i[0][3])
    plt.show()
    