import pronouncing
import random

def stress1(word):
    '''This function returns the list of the given word's stress.'''
    result = []
    pronc = pronouncing.phones_for_word(word)
    if pronc != []:
        for char in pronc[0]:
            if char.isdigit():
                result.append(str(int(char) % 2))
    return result

def rhyme(word):
    '''Return the rhyme of a given word if the last syllable is stressed.'''
    rym = None
    pronc = pronouncing.phones_for_word(word)
    if pronc != []:
        if stress1(word)[-1] == '1':
            rym = pronc[0].split()[-1]
    return rym
    
def words_same_rhyme(rym, wd_lst):
    '''Return all words in 'wd_lst' that have the same rhyme with 'word'.'''
    word_set = set()
    for wd in wd_lst:
        if rhyme(wd) == rym:
            word_set.add(wd)
    return word_set

def fourteen_end_words(wd_lst):
    rym_set = set()
    for wd in wd_lst:
        if rhyme(wd) != None:
            if len(words_same_rhyme(rhyme(wd), wd_lst)) >= 2:
                rym_set.add(rhyme(wd))
    # randomly choose 7 different rhymes
    rym7_lst = random.sample(rym_set, 7)
    # randomly choose 14 last words that their rhyme pattern is abab, cdcd, efef, gg                                  
    end_words = 14 * ['']
    for i in range(3):
        for j in range(2):
            two_words = random.sample(words_same_rhyme(rym7_lst[i*2+j], wd_lst), 2)
            end_words[i*4+j] = two_words[0]
            end_words[i*4+j+2] = two_words[1]
    two_words = random.sample(words_same_rhyme(rym7_lst[6], wd_lst),2)
    end_words[12] = two_words[0]
    end_words[13] = two_words[1]
    return end_words
