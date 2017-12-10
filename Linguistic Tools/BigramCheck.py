# python3 BigramCheck.py /Users/lucasvergeest/git/tts-lexicon/nl-NL/nl-NL_lexicon_nashville_only_Dutch > BigramCheck.txt

import re
import sys
import collections

bigrams = []

with open(sys.argv[1]) as lexicon:
    for line in lexicon:
        transcription = line[line.find(' ')+1:-1]
        transcription = re.sub(':','',transcription)
        transcription = transcription.split()
        for phone in range(len(transcription)-1):
            bigram = transcription[phone] + transcription[phone+1]
            bigrams.append(bigram)

counter = collections.Counter(bigrams)

print(counter)