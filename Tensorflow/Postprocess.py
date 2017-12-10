'''
python Postprocess.py RawResult.txt > Result.csv
'''
# Postprocessing of the data for the Otto Kaggle competition to ensure that the output is converted to the requested format

from __future__ import print_function
import re
import operator
import sys

n = 0
highestvalue = 0
matrix2 = []

print('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9')

with open(sys.argv[1], 'r') as matrix:
    while True:
        line = matrix.readline()
        line2 = matrix.readline()
        line3 = matrix.readline()
        if not line3: break
        line4 = line + line2 + line3
        line5 = []
        line6 = []
        line4 = re.findall('[0-9].*?e[\-\+][0-9][0-9]', line4)
        n+=1
        line4.insert(0,n)
        
        #Print result                
        for element in line4[0:9]:
            print((str(element) + ','), end='')
        print(line4[9])
