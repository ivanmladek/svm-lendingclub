import pandas as pd
import numpy as np
from collections import defaultdict

def dameraulevenshtein(seq1, seq2):
    """
    http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/
    Calculate the Damerau-Levenshtein distance between sequences.

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]

def read_zips_into_states(filename):
    print "Reading zips"
    states_zip = defaultdict(list)
    zips = pd.read_csv(filename)
    for record in zips.values:
        state= record[5]
        states_zip[state].append(record)
    return states_zip

def find_match_in_state(state, city, zip_state, alternate=False):
    """
    Find the city in the state that best matches
    """
    all_cities = zip_state[state]
    distances = list()
    for z,typ,prim_city,acc_city,un_city,_,_,_,_,_,_,_,_,_,pop,_  in all_cities:
        #First letters have to match
        #TODO Large cities span multiple zipcodes,
        #need to return all zipcodes
        if city[:1].lower() == prim_city[:1].lower():
            d = float(dameraulevenshtein(
                    city.lower(),
                    prim_city.lower()))
            distances.append([z,pop,prim_city, d])
            if d == 0.0:
                break
        #loop through acceptable cities if alternate is True
        if isinstance(acc_city, str) and alternate == True:
            for acc in acc_city.split(","):
                distances.append([z,pop,acc, float(dameraulevenshtein(
                                city.lower(),
                                acc.lower()))])
    nearest_match = min(distances, key = lambda x: x[3])
    return nearest_match

def main():
    zip_state = read_zips_into_states("zip_code_database_standard.csv")
    lending_corpus= pd.read_csv("../LoanStatsNew.csv")
    column_names = lending_corpus.columns
    for i,cc in enumerate(column_names):
        print i,cc
    for c in lending_corpus.values[0:1000]:
        state = c[27]
        city = c[26]
        nearest_city_match = find_match_in_state(state, city, zip_state)
        print state, city, nearest_city_match

if __name__ == '__main__':
    main()
