import pandas as pd
import numpy as np
from collections import defaultdict
from svm_lending import parser_options

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

    def __init__(self):
        self.parser_options = parser_options

    def read_zips_into_states(self, filename_zip, filename_census):
        states_zip = defaultdict(list)
        print "Reading zips"
        z = pd.read_csv(filename_zip)
        print "Reading census"
        c = pd.read_csv(filename_census)
        #Rename census columns
        rename_c = c.rename(columns={'GEO.id2': 'zip', '$b': 'b'})
        print 'Merging Census with zip'
        census_zip = pd.merge(rename_c,z)
        for row_index, row in census_zip.iterrows():
            states_zip[row['state']].append(row)
        return states_zip

    def find_match_in_state(self, state,
                            city, zip_state, alternate=False):
        """
        Find the city in the state that best matches
        """
        all_cities = zip_state[state]
        distances = list()
        for c_info in all_cities:
            prim_city = c_info['primary_city']
        #First letters have to match
            if city[:1].lower() == prim_city[:1].lower():
                d = float(dameraulevenshtein(
                        city.lower(),
                        prim_city.lower()))
                distances.append([c_info, d])
                #TODO Large cities span multiple zipcodes,
                #need to return all zipcodes
                if d == 0.0:
                    break
            ##loop through acceptable cities if alternate is True
            #if isinstance(acc_city, str) and alternate == True:
            #    for acc in acc_city.split(","):
            #        distances.append([z,pop,acc, float(dameraulevenshtein(
            #                        city.lower(),
            #                        acc.lower()))])
        nearest_match = min(distances, key = lambda x: x[1])
        return nearest_match

    def process_file(self,in_file):
        """
        Geocode file and return a new Pandas DataFrame which
        is geocoded.
        """
        zip_state = self.read_zips_into_states("zip_code_database.csv",
                                               "ACS_11_5YR_DP03_with_ann.csv")
        lending_corpus = self.parser_options[in_file](in_file)
        #Read file using the right parser
        geocoded = list()
        for _, row in lending_corpus.iterrows():
            nearest_city_match = self.find_match_in_state(
                row['addr_state'],
                row['addr_city'], zip_state)
            #Merge the credit info with the census data
            new_row = row.append(nearest_city_match[0])
            print new_row
            geocoded = geocoded.append(new_row)
        #http://stackoverflow.com/questions/13653030/how-do-i-pass-a-list-of-series-to-a-pandas-dataframe
        df_geocoded = PD.concat(geocoded, keys = [s.name for s in new_row])
        print geocoded
        return geocoded

if __name__ == '__main__':
    g = Geocode()
    geocoded = g.process_file("InFunding2StatsNew.csv")
    main()
