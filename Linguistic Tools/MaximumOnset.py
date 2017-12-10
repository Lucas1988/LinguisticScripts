# python3 /Users/lucasvergeest/git/scripts/MaximumOnset.py nl-NL_lexicon_nashville_only_Dutch.txt > nl-NL_lexicon_nashville_only_Dutch
# This program tries to apply the maximum onset principle (MOP) to a given annotated lexicon, without violating other phonotactic rules.

import sys
import re
changes = 0

# Open file and read in words and syllables
with open(sys.argv[1]) as lines:
    for line in lines:
        word = line[:line.find(' ')]
        transcription = line[line.find(' ')+1:-1]
        syllables = transcription.split(' . ')
        #syllables = [syllable.replace(':', '') for syllable in syllables]
        
        # Find the potential coda for each syllable
        for i in range (len(syllables)-1):
            coda = re.search('((p|b|t|d|k|g|f|v|s|z|x|h|m|n|r|w|j|l|G|Z|S|N|V|H|D|nK)\s?)+$', syllables[i])
            
            # If coda is present: move the last phone of the coda to the beginning of the onset of the next syllable
            if coda:           
                newsyllable = (coda.group())[-1:] + ' ' + syllables[i+1]
                onset = re.search('^((p|b|t|d|k|g|f|v|s|z|x|h|m|n|r|w|j|l|G|Z|S|N|V|H|D|nK)\s?)+', newsyllable)
                
                if onset:
                    testonset = onset.group().replace(' ', '')
                
                # Check if no phonotactic rules are violated with the displacement
                if re.search('^(s(f|Hr?|j|kr?|l|m|n|p[lr]?|tr?|w|V)?|S|Z|f[lnr]?|G[lnr]?|g[lnr]?|b[lr]?|d[rwV]?|v[lr]?|z[wV]?|h|r|j|k[slnrwV]?|l|m|n|p[jlr]?|t[jrswV]?|wr?|Vr?)$', testonset) is not None:
                    syllables[i] = syllables[i][:-2]
                    syllables[i+1] = newsyllable
                    changes += 1
                    
                    # Display changes in specific words
                    #print(word, syllables)
                    
        # Output the results
        print(word, (' . '.join(str(p) for p in syllables)))
        
    #print(str(changes) + ' onset changes were made in the data.')