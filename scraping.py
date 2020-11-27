import sys
import time
import datetime

from unidecode import unidecode
import pandas as pd
import numpy as np

import os

try:
    from Bio import Entrez
except:
    pass

try:
    from scholarly import scholarly
except:
    pass

try:
     import requests
     #import feedparser
except:
    pass



class abstractScraper():

    """ Basic scraper for downloading abstracts"""
    
    def __init__(self, keywords, database='pubmed', email=None, retmax=1000, outPath='./', verbose=False):

        
        """ 
        Args:
            keywords (list of str): list of keywords to investigate.
            database (list or str): string or list of strings of databases to scrape
                        'pubmed', 'biorxiv' and 'scholar' are currently implemented (default 'pubmed').
            email (str): email address for PubMed retrival. 
            retmax (int): total number of IDs to retrieve per database.
            outPath (str): path to folder where files will be saved.
            verbose (bool): activate verobse option (search timing).
        """

        self.email = email
        self.keywords = ' '.join(keywords)
        self.database = database
        
        if not isinstance(database, list):
            self.database=[self.database]

        if 'pubmed' in self.database and 'Bio' not in sys.modules:
            warnings.warn('Bio.Entrez not found, skipping PubMed search')
            self.database=[x for x in self.database if x!='pubmed']
        
        if 'biorxiv' in self.database and 'requests' not in sys.modules:
            warnings.warn('Requests not found, skipping Biorxiv search')
            self.database=[x for x in self.database if x!='biorxiv']

        if 'scholar' in self.database and 'scholarly' not in sys.modules:
            warnings.warn('Scholarly not found, skipping Google Scholar search')
            self.database=[x for x in self.database if x!='scholar']

        self.retmax = retmax

        self.chunkSize = 50
        
        self.outPath = outPath
        if not os.path.exists(self.outPath):
            os.makedirs(self.outPath)

        self.verbose = verbose

        self.found = []


    def searchEntrez(self):

        """ Search query on PubMed with Entrez. 

        Returns:
            (list) list of paper IDs found by the scraper.

        """

        Entrez.email = self.email
        
        handle = Entrez.esearch(db='pubmed', 
                                sort='relevant', 
                                retmax=str(self.retmax),
                                retstart = '10',
                                retmode='xml', 
                                #reldate = '3650',
                                term=self.keywords)
        
        return Entrez.read(handle)['IdList']

    def fetchDetails(self, idsList):

        """ Fetch details for a list of publication IDs. 

        Args:
            isdList (list of str): list of IDs.

        Returns:
            (list): details for each publication in input.

        """

        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=','.join(idsList))
        return Entrez.read(handle)


    def scrapePubMed(self):

        """ Download papers information from PubMed. """

        ids = self.searchEntrez()
 
        for chunkIx in range(0, len(ids), self.chunkSize):
            chunk = ids[chunkIx : chunkIx + self.chunkSize]

            try:
                papers = self.fetchDetails(chunk)       
                for i, paper in enumerate(papers['PubmedArticle']):

                    self.found.append([unidecode(paper['MedlineCitation']['Article']['ArticleTitle']),
                                          unidecode(paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]),
                                          unidecode(paper['MedlineCitation']['PMID']),
                                          unidecode(paper['MedlineCitation']['Article']['Journal']['Title']),
                                          unidecode(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'])])

                    if 'Month' in paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']:
                        self.found[-1].append(unidecode(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Month']))
                    else:
                        self.found[-1].append(np.nan)

                    auts = []
                    for aut in paper['MedlineCitation']['Article']['AuthorList']:
                            auts.append(aut['Initials']+' '+aut['LastName'])

                    self.found[-1].append(auts)

            except:
                pass


    def scrapeBiorxivFeed(self):

        """ DEPRECATED: download papers information from biorixiv with feeds (can only access last 30 abstracts).  """

        feed = feedparser.parse('http://connect.biorxiv.org/biorxiv_xml.php?subject=%s'\
                                % ('biochemistry+bioinformatics+biophysics+cancer_biology+cell_biology+\
                                    clinical_trials+genetics+genomics+immunology+molecular_biology+pathology'))

        for i in range(len(feed.entries)):
            if any(key.lower() in feed.entries[i].summary.lower() for key in self.keywords.split()):
                self.found.append([feed.entries[i].title.replace('\n', " "),
                                    feed.entries[i].summary.replace('\n', " "),
                                    feed.entries[i].link.replace('\n', " "),
                                    'bioRxiv',
                                    feed.entries[i].updated[:4],
                                    feed.entries[i].updated[5:7],
                                    feed.entries[i].author
                                    ])


    def scrapeBiorxiv(self):

        """ Download papers information from biorixiv. """

        # WARNING: not a great implementation, particularly inefficient.

        collection = ['','']
        totalpapers = 0
        chunkIx = 0

        while len(collection)!=0 and chunkIx<100000:


            if chunkIx > 10000 and totalpapers < chunkIx/100:
                print('WARNING: too few papers found, bioRxiv search stopped.')
                break

            if totalpapers >= self.retmax:
                break

            collection = requests.get('https://api.biorxiv.org/details/biorxiv/2015-01-01/{}/{:d}'\
                                .format(datetime.date.today().strftime('%Y-%m-%d'),chunkIx)).json()['collection']
            
            chunkIx += 100

            for paper in collection:

                #print(paper['abstract'])
                if any(key.lower() in paper['abstract'].lower() for key in self.keywords.split()):

                    self.found.append([paper['title'].replace('\n', " "),
                                        paper['abstract'].replace('\n', " "),
                                        paper['doi'].replace('\n', " "),
                                        'bioRxiv',
                                        paper['date'][:4],
                                        paper['date'][5:7],
                                        paper['authors']
                                        ])
                    totalpapers += 1

        print(chunkIx, totalpapers)

    def scrapeScholar(self):

        """ Download papers information from Scholar """

        for i, paper in enumerate(scholarly.search_pubs(self.keywords)):

                if 'abstract' in paper.bib:
                    self.found.append([paper.bib['title'],
                                        paper.bib['abstract'],
                                        paper.bib['url'],
                                        paper.bib['venue'],
                                        paper.bib['year'],
                                        '',
                                        paper.bib['author']
                                        ])

                if i == self.retmax -1:

                    break


    def scrape(self):

        """ Run a full search on the databases of choice. """ 

        start = time.time()

        """ PubMed """
    
        if 'pubmed' in self.database:

            self.scrapePubMed()
          
        """ Biorxiv """

        if 'biorxiv' in self.database:

            self.scrapeBiorxiv()

        """ Google Scholar """

        if 'scholar' in self.database:

            self.scrapeScholar()


        self.found=pd.DataFrame(self.found, columns=['title','abstract','PMID/URL/DOI','journal','year','month','author']).astype(str).fillna('')
        self.found=self.found.drop_duplicates(subset='title')

        end = time.time()

        if self.verbose:

            print('Titles saved {:d}'.format(self.found.shape[0]))
            print('Time elapsed: {}'.format(datetime.timedelta(seconds=round(end - start))))


    def save(self, format='hdf5'):


        """ Save the papers to disk. 

        Args:
            format (str): the output file format (only hdf5 and csv are available, default hdf5).

        """

        if format == 'hdf5':
            self.found.to_hdf(os.path.join(self.outPath,'scraped.h5'), key='df')
        elif format == 'csv':
            self.found.to_csv(os.path.join(self.outPath,'scraped.csv'))
        else:
            print('Error: '+str(format)+' is an invalid format (only hdf5 and csv are currently available).')

  