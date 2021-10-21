from collections import defaultdict
from nltk.corpus import WordNetCorpusReader
from os.path import exists

#made by Alex Moreo at https://stackoverflow.com/questions/21902411/how-to-get-domain-of-words-using-wordnet-in-python
class WordNetDomains:
    def __init__(self, wordnet_home):
        #This class assumes you have downloaded WordNet2.0 and WordNetDomains and that they are on the same data home.
        #assert exists(f'{wordnet_home}/WordNet-2.0'), f'error: missing WordNet-2.0 in {wordnet_home}'
        #assert exists(f'{wordnet_home}/wn-domains-3.2'), f'error: missing WordNetDomains in {wordnet_home}'

        # load WordNet2.0
        self.wn = WordNetCorpusReader(f'{wordnet_home}/WordNet-2.0/dict', 'WordNet-2.0/dict')

        # load WordNetDomains (based on https://stackoverflow.com/a/21904027/8759307)
        self.domain2synsets = defaultdict(list)
        self.synset2domains = defaultdict(list)
        for i in open(f'{wordnet_home}/wn-domains-3.2/wn-domains-3.2-20070223', 'r'):
            ssid, doms = i.strip().split('\t')
            doms = doms.split()
            self.synset2domains[ssid] = doms
            for d in doms:
                self.domain2synsets[d].append(ssid)

    def get_domains(self, word, pos=None):
        word_synsets = self.wn.synsets(word, pos=pos)
        domains = []
        for synset in word_synsets:
            domains.extend(self.get_domains_from_synset(synset))
        return set(domains)

    def get_domains_from_synset(self, synset):
        return self.synset2domains.get(self._askey_from_synset(synset), set())

    def get_synsets(self, domain):
        return [self._synset_from_key(key) for key in self.domain2synsets.get(domain, [])]

    def get_all_domains(self):
        return set(self.domain2synsets.keys())

    def _synset_from_key(self, key):
        offset, pos = key.split('-')
        return self.wn.synset_from_pos_and_offset(pos, int(offset))

    def _askey_from_synset(self, synset):
        return self._askey_from_offset_pos(synset.offset(), synset.pos())

    def _askey_from_offset_pos(self, offset, pos):
        return str(offset).zfill(8) + "-" + pos
