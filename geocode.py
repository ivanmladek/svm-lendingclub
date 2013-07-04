import pandas as pd
import math
import numpy as np
from collections import defaultdict
from svm_lending import parser_options
from datetime import datetime
import re

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

class Geocode():
    """
    Geocode input DataFrame and output geocoded
    DataFarme.
    """
    def __init__(self):
        self.parser_options = parser_options

    def read_zips_into_states(self, filename_zip, filename_census):
        """
        Merge census and zip info.
        """
        zipmap = defaultdict(list)
        print "Reading zips"
        z = pd.read_csv(filename_zip)
        print "Reading census"
        c = pd.read_csv(filename_census)
        #Rename census columns
        rename_c = c.rename(columns={'GEO.id2': 'zip', '$b': 'b'})
        print 'Merging Census with zip'
        census_zip = pd.merge(rename_c,z)
        #TODO add list into states dict
        for row_index, row in census_zip.iterrows():
            zipmap[row['state']].append(row)
        return zipmap

    def find_match_in_state(self, state,
                            city, zip_state, alternate=False):
        """
        Find the city in the state that best matches
        the primary_city in the state.
        """
        #TODO Convert zip_state to numpy array for searching for the
        #nearest city and state
        state_zip_array =  np.array([c['primary_city'].lower() for c in zip_state[state]])
        city_ix =  [ i for i, word in enumerate(state_zip_array) if city.lower() in word.lower() ]

        #regex = re.compile(".*("+city.lower()+").*")
        #found_cities =  [m.group(0) for l in state_zip_array for m in [regex.search(l)] if m]
        #print  city, state, found_cities

        #nearest_match_ix = np.where(np.any(state_zip_array==city.lower(),axis=0))
        #print city, state, state_zip_array[nearest_match_ix,:]
        #try:
            #nearest_match = zip_state[state][city.lower()]
        #except:
        #    distances = list()
        #    for k, c_info in zip_state[state].iteritems():
        #        prim_city = c_info['primary_city']
        #        d = float(dameraulevenshtein(
        #                city.lower(),
        #                prim_city.lower()))
        #        distances.append([c_info, d])
        #        if d == 0.0:
        #            break
        #    nearest_match = min(distances, key = lambda x: x[1])[0]
        return 0 #nearest_match

    def process_file(self,in_file):
        """
        Geocode file and return a new Pandas DataFrame which
        is geocoded.
        """
        print type(in_file)
        zip_state = self.read_zips_into_states("zip_code_database.csv",
                                               "ACS_11_5YR_DP03_with_ann.csv")
        #TODO Don't geocode already geocoded files

        #Read file using the right parser
        if type(in_file) == str:
            lending_corpus = self.parser_options['LoanS'](in_file)
        else:
            lending_corpus = self.parser_options['InFun'](in_file)
            in_file = 'current_offers_' + \
                datetime.now().date().strftime('%Y-%m-%d-%h')

        geo_df = pd.DataFrame()
        lc = len(lending_corpus)
        for i,(_, row) in enumerate(lending_corpus.iterrows()):
            if (i/100. == (math.floor(i/100.))): print i, lc, float(float(i)/float(lc))
            nearest_city_match = self.find_match_in_state(
                row['addr_state'],
                row['addr_city'], zip_state)
            #Merge the credit info with the census data
            #new_row = row.append(nearest_city_match)
            #new_df = pd.DataFrame(new_row).T
            #geo_df = geo_df.append(new_df)
        #Write to a file
        geo_df.to_csv(in_file+'geo',sep=',',
                          encoding='utf-8')
        print geo_df
        return geo_df

if __name__ == '__main__':
    g = Geocode()
    geocoded = g.process_file("InFunding2StatsNew.csv")
