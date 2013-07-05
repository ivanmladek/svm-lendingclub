import pandas as pd
import math
import numpy as np
from collections import defaultdict
from svm_lending import parser_options
from datetime import datetime
import re
import jellyfish

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

    def find_match_in_state(self,i, lc, state,
                            city, zip_state, alternate=False):
        """
        Find the city in the state that best matches
        the primary_city in the state.
        """
        if i/100. == np.floor(i/100.): print i,lc
        state_zip_array =  np.array([c['primary_city'].lower()
                                     for c in zip_state[state]])
        #Find litearal match
        city_ix =  [ i for i, word in enumerate(state_zip_array)
                     if city.lower() in word.lower() ]
        if city_ix == []:
            try:
                distances = ([c, float(jellyfish.levenshtein_distance(
                                c['primary_city'].lower(),
                                city.lower()))] for c in zip_state[state])
                nearest_match = min(distances, key = lambda x: x[1])[0]
            except:
                #Sometimes there really is no match at all, return a
                #dummy city
                print state, zip_state[state]
                nearest_match = zip_state['AK'][0]
        else:
            #Pick the first entry of al matching cities
            #TODO Find a way how to average all entries from let's say
            #New York
            nearest_match = zip_state[state][city_ix[0]]
        #print city, state, nearest_match['primary_city']
        return nearest_match

    def geocode_df(self,lending_corpus, in_file):
        #From http://factfinder2.census.gov/faces/tableservices/\
            #jsf/pages/productview.xhtml?pid=ACS_11_5YR_DP03&prodType=table
        zip_state = self.read_zips_into_states("zip_code_database.csv",
                                               "ACS_11_5YR_DP03_with_ann.csv")
        #TODO Don't geocode already geocoded files
        lc = len(lending_corpus)
        matched_lending_census = [
            row.append(self.find_match_in_state(row_ix, lc,
                                                row['addr_state'],
                                                row['addr_city'], zip_state))
            for row_ix, row in lending_corpus.iterrows()]
            #Merge the credit info with the census data
        print 'matching new rows'
        #new_row = [n[0].append(n[1]) for n in matched_lending_census]
        geo_df = pd.DataFrame(matched_lending_census)#.T
        #Write to a file
        print 'printing file'
        geo_df.to_csv(in_file+'geo',sep=',',
                          encoding='utf-8')
        return geo_df

    def process_file(self,in_file, geocode=True):
        """
        Geocode file and return a new Pandas DataFrame which
        is geocoded.
        """
        print type(in_file)
        #Read file using the right parser
        print "reading file"
        if type(in_file) == str:
            lending_corpus = self.parser_options['LoanS'](in_file)
        else:
            lending_corpus = self.parser_options['InFun'](in_file)
            in_file = 'current_offers_' + \
                datetime.now().date().strftime('%Y-%m-%d-%h')
        #Only geocode on-demand
        if geocode:
            print 'Geocoding for ',in_file
            geo_df = self.geocode_df(lending_corpus, in_file)
        else:
            print 'Skipping geocoding for ', in_file
            geo_df = lending_corpus
        print geo_df
        return geo_df

if __name__ == '__main__':
    g = Geocode()
    geocoded = g.process_file("InFunding2StatsNew.csv")
