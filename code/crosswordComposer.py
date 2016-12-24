from __future__ import print_function
import copy
import numpy as np
# List of word possibilities for each slot
# Hard-coded words at given rows
# Scan starting puzzle to eliminate invalid possibilities from intersecting columns
# Starting with the first unfillled column pick a word and remove all invalid possibilities for intersecting slots

# Store two lists of answer slots, across and down.
# Specify the start point and lengths of each slot
# Specify intersections with the other list 
def assign(puzzle, acrossOrDown, idx, word):
    puzzle[acrossOrDown][idx] = [word]
    # Check to make sure all prefixes are still valid?

    # eliminate all conflicting words from intersecting lists
    # Make sure we don't have these backwards 
    if acrossOrDown == 'aw':
    # position in across term, position in down term, index in down list
    # index into acrossIntersect is the index into puzzle['ai']
    # acrossIntersect = [((0, 0, 0), (3, 0, 1)), ((0, 3, 0), (3, 3, 1))]
        for interSect in puzzle['ai'][idx]:
            pat = interSect[0]
            pdt = interSect[1]
            idl = interSect[2]
            newWords = []
            total    = 0
            # Check all remaining words in the intersecting list and keep any that are still viable
            for downWord in puzzle['dw'][idl]:
                if word[pat] == downWord[pdt]:
                    newWords.append(downWord)
                total += 1
            puzzle['dw'][idl] = newWords
            # one word left!  make sure it fits in the slot
            if len(newWords) == 1 and total != 1:
                if not assign(puzzle, 'dw', idl, newWords[0]):
                    return False
            # no valid possibilities for this slot (column)
            if len(newWords) == 0:
                return False
    elif acrossOrDown == 'dw':
    # position in down term, position in across term, index in across list
    # index into acrossIntersect is the index into puzzle['di']
    # downIntersect   = [((0, 0, 0), (3, 0, 1)), ((0, 3, 0), (3, 3, 1))]
        for interSect in puzzle['di'][idx]:
            pdt = interSect[0]
            pat = interSect[1]
            ial = interSect[2]
            newWords = []
            total    = 0
            # Check all remaining words in the intersecting list and keep any that are still viable
            for acrossWord in puzzle['aw'][ial]:
                if word[pdt] == acrossWord[pat]:
                    newWords.append(acrossWord)
                total += 1
            puzzle['aw'][ial] = newWords
            # one word left!  make sure it fits in the slot
            if len(newWords) == 1 and total != 1:
                if not assign(puzzle, 'aw', ial, newWords[0]):
                    return False
            # no valid possibilities for this slot (column)
            if len(newWords) == 0:
                return False
    else:
        raise Exception("WTF")

    return puzzle 

def search(puzzle):
    if puzzle is False:
        return False

    # Detect completion
    if all(len(wordlist) == 1 for wordlist in puzzle['aw']):
        if all(len(wordlist) == 1 for wordlist in puzzle['dw']):
            return puzzle

    # Try progressing from the upper left to the lower right, alternating rows and columns

    # Get the length of the shortest length list of possibilities, and that list itself
    # Assume it's the down, correct if it isn't
    acrossOrDown        = 'dw'
    downFinished        = False
    try:
        n, wordList, idx    = min( (len(puzzle['dw'][i]), puzzle['dw'][i], i) for i in range(len(puzzle['dw'])) if len(puzzle['dw'][i]) > 1 )
    except ValueError:
    # This happens when we've assigned all down words
        downFinished = True
        pass

    try:
        an, awordList, aidx = min( (len(puzzle['aw'][i]), puzzle['aw'][i], i) for i in range(len(puzzle['aw'])) if len(puzzle['aw'][i]) > 1 )
    except ValueError:
    # This happens when we've assigned all across words
        pass

    #n,s = min( (len(s), s, idx) for s in values if len(s) > 1 )
    if downFinished or an < n:
        acrossOrDown = 'aw'
        idx          = aidx
        wordList     = awordList

    acrossSlotsFilled = len([w for w in puzzle['aw'] if len(w) == 1])
    downSlotsFilled   = len([w for w in puzzle['aw'] if len(w) == 1])

    print("%d filled of %d" %(acrossSlotsFilled+downSlotsFilled, len(puzzle['aw'])+len(puzzle['dw'])))

    return some(search(assign(copy.deepcopy(puzzle), acrossOrDown, idx, word)) for word in wordList)

def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False

# Takes:
#   two lists of (x,y) points representing the squares occupied by the
#   horizontal (across) word and the squares occupied by the vertical (down) word
# Returns:
#   False if the words have no letters in common (i.e. do not intersect) 
#   A tuple of: the slot in the across word and the slot in the down word where the intersection occurs
#   Note that this is *NOT NECESSARILY* the same as the (x,y) coordinates of the intersection
def wordsIntersect(acrossWord, downWord):
    # Get the column (x-value) of the down word and the row (y-value) of the across word
    intersectionPoint = (downWord[0][0], acrossWord[0][1])
    # If this point is in both words they intersect
    try:
        aSlot = acrossWord.index(intersectionPoint)
    except ValueError:
        return False
    try:
        dSlot = downWord.index(intersectionPoint)
    except ValueError:
        return False

    return (aSlot,dSlot)

# Takes: a (startpoint, endpoint) pair
# Returns: a list points for that word
def expandWord(startEndPair):
    start  = startEndPair[0]
    end    = startEndPair[1]
    #print("Start:",start,"End:",end)
    points = []
    if end[0] - start[0] > 0:
        for i in range(start[0], end[0]+1):
            points.append((i,end[1]))

    if end[1] - start[1] > 0:
        for j in range(start[1], end[1]+1):
            points.append((start[0],j))
    return points

def constructIntersections(across, down):
    acrossIntersect = []
    downIntersect   = []
    for i in range(0,len(across)):
        aw = expandWord(across[i])
        tmpList = []
        for j in range(0, len(down)):
            dw = expandWord(down[j])
            intersect = wordsIntersect(aw, dw)
            if intersect != False:
                tmpList.append((intersect[0], intersect[1], j))

        acrossIntersect.append(tuple(tmpList))

    for i in range(0,len(down)):
        dw = expandWord(down[i])
        tmpList = []
        for j in range(0, len(across)):
            aw = expandWord(across[j])
            intersect = wordsIntersect(aw, dw)
            if intersect != False:
                tmpList.append((intersect[1], intersect[0], j))

        downIntersect.append(tuple(tmpList))
                
    return (acrossIntersect, downIntersect)

def initializeWordLists(dictionaryFile, acrossSlots, downSlots):
    wordsBySize = {}
    acrossWords = []
    downWords   = []
    with open(dictionaryFile) as f:
        for line in f:
            word = line.rstrip().lower()
            try:
                wordsBySize[len(word)].append(word)
            except KeyError:
                wordsBySize[len(word)] = [word]

    for w in acrossSlots:
        size = len(expandWord(w))
        acrossWords.append(wordsBySize[size])

    for w in downSlots:
        size = len(expandWord(w))
        downWords.append(wordsBySize[size])

    return (acrossWords, downWords)

def slotsFromInvalidSquares(matrix, flip=True):
    across = []
    inSlot = False
    for i in range(0,matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if j != matrix.shape[1]-1 and not(inSlot) and matrix[i,j] == 0:
                if flip:
                    start = (j,i) # flipped because I'm weird
                else:
                    start = (i,j)
                inSlot = True
            if matrix[i,j] != 0 and inSlot:
                if flip:
                    end = (j-1,i)  # flipped because I'm weird
                else:
                    end = (i,j-1)
                inSlot = False
                if start != end:
                    across.append((start, end))
            if inSlot and j == matrix.shape[1]-1:
                if flip:
                    end = (j,i)  # flipped because I'm weird
                else:
                    end = (i,j)
                inSlot = False
                if start != end:
                    across.append((start, end))
    return across

def slotsFromInvalidSquaresMatrix(matrix):
    across = slotsFromInvalidSquares(matrix)
    down   = slotsFromInvalidSquares(matrix.T, flip=False)
    return (across,down)
 
#def setFixedWords([('vrbo', 2), ('candace', 11), ('everything', 12), ('dan', 18)], acrossWords)
def setFixedWords(wordList, fillArray):
    for word,idx in wordList:
        if len(fillArray[idx][0]) == len(word):
            fillArray[idx] = [word]
        else:
            raise Exception("Specified word does not match slot length: %s at %d %d != %d" %(word, idx, len(word), len(fillArray[idx][0])))

if __name__ == '__main__':
    puzzleSize = 4
    dictionaryFile = '/Users/deronne/Downloads/linuxwords.txt'

#grid = [words]*puzzleSize*2 # square puzzles only for now

# These are the slot definitions for the upper 6x6 from the horn clauses in crosswords
# paper
#across = [((0,0),(3,0)),((0,1),(3,1)),((0,2),(3,2)),((0,3),(2,3)),((4,3),(5,3)),((0,4),(5,4)),((2,5),(5,5))]
#down   = [((0,0),(0,4)),((1,0),(1,4)),((2,0),(2,5)),((3,0),(3,2)),((3,4),(3,5)),((4,3),(4,5)),((5,0),(5,5))]

#matrix = np.zeros((7,7), dtype=int)
# Specify these as row then column and the code takes care of the rest
#matrix[0][4] = 1
#matrix[1][4] = 1
#matrix[2][4] = 1
#matrix[3][3] = 1
#matrix[5][0] = 1
#matrix[5][1] = 1
#matrix[5][6] = 1
#matrix[6][5] = 1

    matrix = np.zeros((15,15), dtype=int)
    matrix[0,4] = 1
    matrix[1,4] = 1
    matrix[2,4] = 1
    matrix[0,10] = 1
    matrix[1,10] = 1
    matrix[2,10] = 1
    matrix[3,3] = 1
    matrix[4,7] = 1
    matrix[4,8] = 1
    matrix[4,12] = 1
    matrix[4,13] = 1
    matrix[4,14] = 1
    matrix[5,0] = 1
    matrix[5,1] = 1
    matrix[5,6] = 1
    matrix[6,5] = 1
    matrix[6,10] = 1
    matrix[7,3] = 1
    matrix[7,11] = 1

    matrix = matrix +np.rot90(matrix, 2)

    across, down = slotsFromInvalidSquaresMatrix(matrix)
    acrossIntersect, downIntersect = constructIntersections(across, down)
    acrossWords, downWords         = initializeWordLists(dictionaryFile, across, down)

# Specify fixed words here (after intersections and slot lengths have been determined)

    setFixedWords([('vrbo', 2), ('candace', 11), ('dan', 18)], acrossWords)
    setFixedWords([('prince', 13), ('normandy', 14), ('everything', 32)], downWords)

    puzzle = {'aw':acrossWords, 'dw':downWords, 'ai':acrossIntersect, 'di':downIntersect}
    result = search(puzzle)
    print(result['aw'])
    print(result['dw'])
    exit()

#-------------------------------------------------------------------------------- 
# Test case: 4x4 square with a 2x2 blackout in the center
#-------------------------------------------------------------------------------- 
# x,y,length
    across = [(0,0,4), (0,3,4)]
# x,y,length
    down   = [(0,0,4), (3,0,4)]
# position in across term, position in down term, index in down list
    acrossIntersect = [((0, 0, 0), (3, 0, 1)), ((0, 3, 0), (3, 3, 1))]
# position in down term, position in across term, index in across list
    downIntersect   = [((0, 0, 0), (3, 0, 1)), ((0, 3, 0), (3, 3, 1))]

    acrossWords = [words]*len(across)
    downWords   = [words]*len(down)
    downWords[0] = ['paul']
    puzzle = {'a': across, 'd': down, 'aw':acrossWords, 'dw':downWords, 'ai':acrossIntersect, 'di':downIntersect}
    result = search(puzzle)

    print(result)
#for i in range(puzzleSize, puzzleSize*2):
#    print(result[i][0])

