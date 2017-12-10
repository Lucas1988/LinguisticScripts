# python3 /Users/lucasvergeest/git/scripts/MissingWords.py

from lipservice import Lipservice
import os
import sys
import re

Replacements = [['.','\_'],['N','nK'],['G','x'],['J','E\&i'],['Y','\^\&y'],['130','er'],['133','Ar'],['138','Er'],['141','Ir'],['148','e\+'],['144','E\:'],['149','Or'],['152','yr'],['155','e\+r'],['160','ar'],['161','ir'],['162','or'],['163','ur'],['q','\$r'],['Q','\^r'],['@','A\&u'],['H','x'],['$','\%'],['^','\#'],['+','\~'],['','\'2'],['','\?']]

voice = 'klaar_021'
ls = Lipservice()
ls.list_voices()

dictionary = {}

with open('/Users/lucasvergeest/Documents/MissingWords.txt') as MissingWords:
    for line in MissingWords:
        line = line.lower()
        dictionary[line] = None

print('Length of dictionary:', len(dictionary))

silent_garbage = []
for k, v in sorted(dictionary.items()):
    trans,_ = ls.translate_text(voice,k)
    
    NashvilleTranscription = trans
    
    for Replacement in Replacements:
        NashvilleTranscription = re.sub(Replacement[1], Replacement[0], NashvilleTranscription)   

    NashvilleTranscription = re.sub(r'\'(.*?[0-9]{3})', r'\1:', NashvilleTranscription)        
    NashvilleTranscription = re.sub(r'\'(.*?[aeiouyAEIOYJQ\@\^\$]\:?)', r'\1:', NashvilleTranscription)
    NashvilleTranscription = re.sub(r'([a-zA-Z\.\:\$\@\^])', r'\1 ', NashvilleTranscription)
    NashvilleTranscription = re.sub(r'(\d+)', r'\1 ', NashvilleTranscription)
    NashvilleTranscription = re.sub(r'(.) :', r'\1:', NashvilleTranscription)
    NashvilleTranscription = re.sub(' $', '', NashvilleTranscription)
    #print(NashvilleTranscription)
    
    if NashvilleTranscription == None:
        NashvilleTranscription = 'ERROR'
        print(k[:-1], NashvilleTranscription)
        silent_garbage.append(k)
    with open('MissingWordsOut.txt', 'a') as out:
        out.write(k[:-1] + ' ' + NashvilleTranscription + '\n')

print('The following did not get a transcription from Lipservice and may be silent characters. Consider cleaning them from the data:')
print(set(silent_garbage))