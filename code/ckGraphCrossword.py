import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import sys
import itertools

def letterOverlaps(wordA, wordB):
    overlaps = []
    for i in range(len(wordA)):
        for j in range(len(wordB)):
            if wordA[i] == wordB[j]:
                overlaps.append((wordA, i, wordB, j))
    return(overlaps)

def wordListLetterOverlaps(wordList):
    allOverlaps = []
    for i in range(len(wordList)-1):
        for j in range(i+1, len(wordList)):
            for overlap in letterOverlaps(wordList[i], wordList[j]):
                allOverlaps.append(overlap)
    return(allOverlaps)    

def pairFirstWord(overlapPair):
    return overlapPair[0]

def pairFirstWordLetterIndex(overlapPair):
    return overlapPair[1]

def pairFirstWordWordIndex(overlapPair):
    return (pairFirstWord(overlapPair), pairFirstWordLetterIndex(overlapPair))

def pairSecondWord(overlapPair):
    return overlapPair[2]

def pairSecondWordLetterIndex(overlapPair):
    return overlapPair[3]

def pairSecondWordWordIndex(overlapPair):
    return (pairSecondWord(overlapPair), pairSecondWordLetterIndex(overlapPair))

def generateGraph(overlaps):
    G = nx.Graph()
    G.add_nodes_from(overlaps)
    for i in range(len(overlaps)-1):
        for j in range(i+1, len(overlaps)):
            pairA = overlaps[i]
            pairB = overlaps[j]
            # Add edges for conflicts using the same letter from the same
            # word in more than one crossing
            if pairFirstWordWordIndex(pairA) == pairFirstWordWordIndex(pairB) or \
                pairFirstWordWordIndex(pairA) == pairSecondWordWordIndex(pairB) or \
                pairSecondWordWordIndex(pairA) == pairFirstWordWordIndex(pairB) or \
                pairSecondWordWordIndex(pairA) == pairSecondWordWordIndex(pairB):
                G.add_edge(pairA, pairB)
            # Add edges for conflicts for crossing a pair of words more than once
            if (pairFirstWord(pairA) == pairFirstWord(pairB) and pairSecondWord(pairA) == pairSecondWord(pairB)) or \
                (pairFirstWord(pairA) == pairSecondWord(pairB) and pairSecondWord(pairA) == pairFirstWord(pairB)):
                G.add_edge(pairA, pairB)
    return(G)

def construct_layout(list_of_overlaps):
    """Construct a layout for the given list of character overlaps. Return
    the layout if feasible. Return None if not possible. 
    """

    # First try horizontal/vertical consistency; construct a graph and
    # then attempt a two-coloring
    G = nx.Graph()
    for pair in list_of_overlaps:
        G.add_edge(pairFirstWord(pair), pairSecondWord(pair))

    try:
        is_vertical = nx.algorithms.bipartite.color(G)
        return is_vertical

    except nx.NetworkXError:
        return None

def ck_maximal_independent_set(G, nodes=None):
    """Return a random maximal independent set guaranteed to contain
    a given set of nodes.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximal
    independent set is an independent set such that it is not possible
    to add a new node and still get an independent set.
    
    Parameters
    ----------
    G : NetworkX graph 

    nodes : list or iterable
       Nodes that must be part of the independent set. This set of nodes
       must be independent.

    Returns
    -------
    indep_nodes : list 
       List of nodes that are part of a maximal independent set.

    Raises
    ------
    NetworkXUnfeasible
       If the nodes in the provided list are not part of the graph or
       do not form an independent set, an exception is raised.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.maximal_independent_set(G) # doctest: +SKIP
    [4, 0, 2]
    >>> nx.maximal_independent_set(G, [1]) # doctest: +SKIP
    [1, 3]
    
    Notes
    -----
    This algorithm does not solve the maximum independent set problem.

    """
    if not nodes:
        nodes = set([list(G.nodes())[0]])
    else:
        nodes = set(nodes)
    if not nodes.issubset(G):
        raise nx.NetworkXUnfeasible(
                "%s is not a subset of the nodes of G" % nodes)
    neighbors = set.union(*[set(G.neighbors(v)) for v in nodes])
    if set.intersection(neighbors, nodes):
        raise nx.NetworkXUnfeasible(
                "%s is not an independent set of G" % nodes)
    indep_nodes = list(nodes)
    available_nodes = set(G.nodes()).difference(neighbors.union(nodes))
    while available_nodes:
        node = list(available_nodes)[0]
        indep_nodes.append(node)
        available_nodes.difference_update(G.neighbors(node) + [node])
    return indep_nodes


def findFeasibleSet(G):
    maximal_overlaps = ck_maximal_independent_set(G)
    print("Maximal independent set of overlaps determined")
    # for i in sorted(maximal_overlaps):
    #     print(i)

    # iterate over all independent sets stemming from a maximal
    # independent set trying to find a feasible set
    feasible_overlaps = None
    layout = None
    for subset_size in range(len(maximal_overlaps),0,-1):                   # Largest to smallest subsets
        for subset in itertools.combinations(maximal_overlaps,subset_size): # All possible subsets of given size
            layout = construct_layout(subset)
            if layout != None:
                print("Found feasible layout with %d overlaps" % (subset_size,))
                feasible_overlaps = subset
                break
        if feasible_overlaps != None:
            break
        
    if feasible_overlaps == None:
        print("No maximal sets could be realized :(")
        return (None, None)
    else:
        print("Feasible with subset:")
        for o in feasible_overlaps:
            print(o)
        print(layout)
        return (layout, feasible_overlaps)

def insertWord(word, matrix, position, orientation, wordLocations):
    first = True
    # Will throw an exception if we run off the end of anything
    #print("Attempting to insert",word,"at",position)
    for i in range(len(word)):
        if orientation == 1: # vertical; increment row
            row, col = position[0]+i, position[1]
        else: # horizontal; increment column
            row, col = position[0], position[1]+i

        if matrix[row][col] == ' ' or matrix[row][col] == word[i]:
            matrix[row][col] = word[i]
            if first:
                wordLocations[word] = ((row,col),orientation)
                first = False
        else:
            matrix[row][col] = 'X'
            for i in range(len(matrix)):
                print("%02d"%i,matrix[i])
            raise Exception("Word collision trying to insert",word,"starting at",position,"colliding at",row,col,"orientation",orientation)

def findPositionForWord(word, feasible_overlaps, layout, wordLocations):
    for pair in feasible_overlaps:
        print("Checking",pair,"for",word)
        intersectingWord = None

        if word == pairFirstWord(pair):
            ourWord          = pairFirstWordWordIndex(pair)
            intersectingWord = pairSecondWordWordIndex(pair)
        if word == pairSecondWord(pair):
            intersectingWord = pairFirstWordWordIndex(pair)
            ourWord          = pairSecondWordWordIndex(pair)

        if intersectingWord and intersectingWord[0] in wordLocations:
            position = wordLocations[intersectingWord[0]][0]
            orientation = wordLocations[intersectingWord[0]][1]

            row,col = position[0],position[1]
            # if orientation is 0 we need to subtract from rows
            if orientation == 0:
                row -= ourWord[1]
                col += intersectingWord[1]
            # if orientation is 1 we need to subtract from cols
            elif orientation == 1:
                col -= ourWord[1]
                row += intersectingWord[1]
            else:
                raise Exception("WTF")
            print("Found location for",word,"intersecting with",intersectingWord[0])
            return (row,col)
        elif intersectingWord:
            print("Found",intersectingWord,"but it wasn't in the grid")
    return None

def buildCrossword(layout, feasible_overlaps, gridSize=20):
    _, longestWord = max([(len(w),w) for w in layout])
    matrix         = []
    for i in range(0, gridSize):
        row = []
        for j in range(0, gridSize):
            row.append(' ')
        matrix.append(row)

    intersections  = {}
    wordLocations  = {}

    # Step through words in increasing length
    words          = sorted(list(layout.keys()), reverse=True)
    firstWord      = True
    for word in words: 
        # Make sure we haven't placed this word already
        if word in layout:
            if firstWord:
                position = (int(gridSize/2),int(gridSize/2))
                firstWord = False
            else:
                position = findPositionForWord(word, feasible_overlaps, layout, wordLocations)

            if not position:
                print("Failed to find position for",word)
                continue

            print("Primary insertion of",word,"at",position)
            insertWord(word, matrix, position, layout[word], wordLocations)

            for pair in feasible_overlaps:
                intersectingWord = None

                if word == pairFirstWord(pair):
                    intersectingWord = pairSecondWordWordIndex(pair)
                    ourWord          = pairFirstWordWordIndex(pair)
                if word == pairSecondWord(pair):
                    intersectingWord = pairFirstWordWordIndex(pair)
                    ourWord          = pairSecondWordWordIndex(pair)

                if intersectingWord:
                    row,col = position[0],position[1]
                    # if orientation is 0 we need to subtract from rows
                    if layout[word] == 0:
                        row -= intersectingWord[1]
                        col += ourWord[1]
                    # if orientation is 1 we need to subtract from cols
                    elif layout[word] == 1:
                        col -= intersectingWord[1]
                        row += ourWord[1]
                    else:
                        raise Exception("WTF")
                    
                    if row < 0 or col < 0:
                        print("Went negative from:",position,"to",(row,col))
                        exit()

                    if intersectingWord[0] in layout:
                        print("Secondary insertion of",intersectingWord[0],"at",(row,col))
                        insertWord(intersectingWord[0], matrix, (row,col), layout[intersectingWord[0]], wordLocations)

                        del(layout[intersectingWord[0]])
            print("Known words:",wordLocations)
            # done with this word
            del(layout[word])
    for row in matrix:
        print(row)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python ckGraphCrossword.py wordList.txt")
        sys.exit(1)

    filename = sys.argv[1]
    wordList = []
    with open(filename) as f:
        wordList = f.read().splitlines()
    
    print("%d words read from %s" % (len(wordList),filename))

    # Over all pairs of words, call letterOverlaps
    overlaps = wordListLetterOverlaps(wordList)
    print("Overlaps constructed")
    #print(overlaps)
    G = generateGraph(overlaps)
    print("Graph constructed")
    # nx.draw_networkx(G);     plt.show()
    layout, feasible_overlaps = findFeasibleSet(G)
    if layout:
        buildCrossword(layout, feasible_overlaps)
