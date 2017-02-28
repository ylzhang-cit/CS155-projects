import time
import nltk
import pronouncing
import numpy as np
import networkx as nx
from HMM import unsupervised_HMM
from rhyme_ends import stress1
from rhyme_ends import fourteen_end_words
import matplotlib.pyplot as plt

class WordStruct:
    '''
    Class implementation of Word Struct.
    '''
    def __init__(self, word):
        self.wd = word
        self.stress = stress1(word)
        self.syllable = len(self.stress)

def readSonnet(fileName):
    """
    This function reads the sonnets, lines, words in the given file.
    Input:
        Each sonnet has 17 lines: number line, 14 lines, 2 empty lines
    Output:
        sonnet_lst: the list of sonnet
        line_lst: the list of lines in all sonnets
        word_set: the set of words appears in the sonnets
    """
    with open(fileName, 'r') as f:
        lines = f.read().splitlines()
    sonnet_lst = []
    line_lst = []
    word_set = set()
    for i, line in enumerate(lines):
        j = i % 17
        if j == 0:
            sonnet = []
        elif (j >= 1) and (j <= 14):
            sonnet.append(line)
            for punc in ",.:;?!()-":
                line = line.replace(punc, ' ')
            words = nltk.word_tokenize(line.lower())
            # words = line.lower().split()
            l = []
            for word in words:
                # add valid words with positive syllables, ignore words with 0 syllable
                if stress1(word) != []:
                    l.append(word)
                    word_set.add(word)
            line_lst.append(l)
        elif j == 15:
            sonnet_lst.append(sonnet)
    return sonnet_lst, line_lst, word_set

def wd2num(line_lst, word_map):
    '''
    This function maps a list of sentences to a list of list of numbers,
    and reverses each list of numbers.'''
    nums_lst = []
    for line in line_lst:
        nums = []
        for wd in line:
            nums.append(word_map[wd])
        nums.reverse()
        nums_lst.append(nums)
    return nums_lst

def num2wd(nums_lst, word_lst):
    '''
    This function maps a list of numbers to a list of sentences, and reverses each sentence.
    '''
    line_lst = []
    for nums in nums_lst:
        line = []
        for num in nums:
            line.append(word_lst[num])
        line.reverse()
        line_lst.append(line)
    return line_lst

def possibleNext(D, x_curr, wd_struct_lst):
    '''
    This function selects the list of possible next numbers,
    such that wd_struct_lst[x_nxt].stress[-1] != wd_struct_lst[x_curr].stress[0].
    Input:
        D:             number of words
        x_curr:        current number
        wd_struct_lst: a list of WordStruct    
    '''
    result = []
    x_curr_stress = wd_struct_lst[x_curr].stress
    for x_nxt in range(D):
        if wd_struct_lst[x_nxt].stress[-1] != x_curr_stress[0]:
            result.append(x_nxt)
    return result

def generateLine(num, A, O, wd_struct_lst):
    '''
    This function will generate a list of numbers that has 10 syllables and good stress pattern.
    Input:
        last_num_lst:  a list of last numbers in each line
        A:             transition matrix
        O:             observation matrix
        wd_struct_lst: a list of WordStruct 
    '''
    nums = [num]
    prob_log = 0
    syllables = wd_struct_lst[num].syllable
    A = np.array(A)
    O = np.array(O)
    L, D = O.shape
    # select the first hidden state
    prob_yx = O[:, num] / np.sum(O[:, num])
    y_curr = np.random.choice(np.arange(L), p=prob_yx)
    x_curr = num
    prob_log += np.log(prob_yx[y_curr])
    # generate y_nxt, x_nxt, add them if total syllables <= 10
    while syllables < 10:
        possible_x_nxt = possibleNext(D, x_curr, wd_struct_lst)
        y_nxt = np.random.choice(np.arange(L), p=A[y_curr])
        x_nxt = np.random.choice(possible_x_nxt, p=O[y_nxt, possible_x_nxt]/np.sum(O[y_nxt, possible_x_nxt]))
        if syllables + wd_struct_lst[x_nxt].syllable <= 10:
            prob_log += np.log(A[y_curr, y_nxt]) + np.log(O[y_nxt, x_nxt])
            nums.append(x_nxt)
            syllables += wd_struct_lst[x_nxt].syllable
            y_curr = y_nxt
            x_curr = x_nxt
    return (nums, prob_log)

def generateSonnet(last_num_lst, A, O, wd_struct_lst):
    '''
    This function generates a sonnet.
    Input:
        last_num_lst:  a list of last numbers in each line
        A:             transition matrix
        O:             observation matrix
        wd_struct_lst: a list of WordStruct
    '''
    nums_lst = []
    for i in range(14):
        nums_max, p_max = [], -10 ** 10
        # choose the line with relatively maximal probability
        for j in range(10):
            nums, p = generateLine(last_num_lst[i], A, O, wd_struct_lst)
            if p > p_max:
                nums_max = nums
                p_max = p
        nums_lst.append(nums_max)
    return nums_lst

def printSonnet(sonnet):
    print('\n')
    print('#' * 75)
    print("{:60}{:15}".format('Sonnet', 'Stress Pattern'))
    print('#' * 75)
    for i, line in enumerate(sonnet):
        stress_pattern = []
        for wd in line:
            stress_pattern += stress1(wd)
        stress_pattern = str.join('', stress_pattern)
        sentence = str.join(' ', line)
        sentence = sentence[0].upper() + sentence[1:]
        if (i == 12 or i == 13):
            sentence = '  ' + sentence
        print("{:60}{:15}".format(sentence, stress_pattern))
    print('\n')


if __name__ == '__main__':
    start = time.time()
    fileName = './project2data/shakespeare.txt'
    sonnet_lst, line_lst, wd_set = readSonnet(fileName)
    wd_lst = list(wd_set)
    # wd_struct_lst[i] is the struct of the i-th word, wd_map[the i-th word] is i
    wd_struct_lst = []
    wd_map = {}
    for i, wd in enumerate(wd_lst):
        wd_map[wd] = i
        wd_struct_lst.append(WordStruct(wd))
    nums_lst = wd2num(line_lst, wd_map)
    print('#' * 75)
    print("Read sonnets from the file '%s'." % fileName)
    print('#' * 75)
    print('The number of sonnets is ', len(sonnet_lst))
    print('The number of lines is ', len(line_lst))
    print('The number of words is ', len(wd_lst))
    #print('The number of last words is ', len(last_wd_lst))
    print('#' * 75)
    print('\n\n')
    # use unsupervised Hidden Markov Model to train all lines in the sonnets
    n_words = len(wd_lst)
    n_states = 8
    n_iters = 100
    HMM = unsupervised_HMM(nums_lst, n_states, n_iters)
    A = np.array(HMM.A)
    O = np.array(HMM.O)
    # Print the transition matrix, observation matrix of first 9 large-prob words
    print("Transition Matrix:")
    print('#' * 75)
    for i in range(len(A)):
        print(''.join("{:<12.3e}".format(A[i][j]) for j in range(len(A[i]))))
    print('\n\n')
    print("Partial Observation Matrix:")
    print('#' * 75)
    for i in range(len(O)):
        index = np.argsort(O[i])[::-1]
        print('%2d:' % i, ''.join("{:12}".format(wd_struct_lst[index[j]].wd) for j in range(10)))
        print('%2d:' % i, ''.join("{:<12.3e}".format(O[i][index[j]]) for j in range(10)))
    print('\n')
    # randomly generate 14 end words that their rhyme pattern is abab, cdcd, efef, gg
    end_words = fourteen_end_words(wd_lst)
    end_nums = [wd_map[wd] for wd in end_words]
    # based on the end_words, generate a 14-line sonnet and print it
    sonnet_num = generateSonnet(end_nums, A, O, wd_struct_lst)
    sonnet_wd = num2wd(sonnet_num, wd_lst)
    printSonnet(sonnet_wd)
    
    # Draw the transition matrix
    DG = nx.DiGraph()
    DG.add_nodes_from(range(n_states))
    # Add edges and their weights
    for i in range(n_states):
        for j in range(n_states):
            if np.abs(A[i,j]) >= 0.05 :
                DG.add_edge(i, j, weight=A[i,j])
    pos = nx.circular_layout(DG)
    nx.draw(DG, pos, with_labels=True)
    edge_labels=dict([((u, v, ), '%.2f'%(d['weight'])) for u, v, d in DG.edges(data=True)])
    nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, label_pos=0.27, font_size=7)
    plt.title('Transition matrix')
    plt.axis('off')
    plt.savefig('shake_transition.png')
    plt.show()

    stop = time.time()
    print('The running time is %0.1fs.' % (stop - start))



    

