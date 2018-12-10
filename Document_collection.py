#!user/bin/env python

# Assignment #1 Information Retrieval Based on SVM
# Author: Shadasha Williams

import gzip
from collections import Counter
import operator
import numpy as np
import math
import re

import xml.etree.ElementTree as etree


class CollectionPipeline:
    """ get a collection of indices for documents, a dictionary that maps terms to an index in vectors, a dictionary representing
     term frequency and inverse document frequency """
    doc_indices = []
    collection_term_indices = {}
    collection_term_frequency = {}
    inverse_doc_frequency = {}

    def __init__(self, terms, stop_words, lower_casing, document_list, search_field, term_weight):
        """
         class  to be initialized with the type of terms being search, the removal of stop words,
        casing preference, a list that includes the path of the files in the collection, the specific
        field to build the document collection from
        :param terms: string to represent the terms to be used in the search engine ie. word forms, lemmas, etc
        :param stop_words: boolean value to represent the removal of stop words
        :param lower_casing: boolean for whether to handle casing
        :param document_list: list of documents to build the search space
        :param search_field: field to be searched ie. the title or the whole document
        :param term_weight: field to represent the term weighting
        """
        self._terms = terms
        self._stop_words = stop_words
        self._lower_casing = lower_casing
        self._document_list = open(document_list, 'r').read().split("\n")
        self._search_field = search_field
        self._term_weight = term_weight
        self._avg_doc_len = 0

    @staticmethod
    def get_doc_terms(text, term_type):
        """ function used  to get specific terms from document text by choosing column """
        escaped_terms = ['"', '&', "'", '>', '<']
        all_terms = []
        col_ind = 1
        if term_type == 'lemmas':
            col_ind = 2
        for line in text.split('\n'):
            if line:
                words = line.split('\t')
                terms = words[col_ind]
                if terms not in escaped_terms:
                    all_terms.append(terms)
        return all_terms

    @staticmethod
    def remove_stopwords_lexical(termlist):
        """ we count the terms for document frequency in the count vector (represented by a dictionary) """
        stop_words = open("stopwords-cs.txt", "r").read().split("\n")
        words_to_remove = set(stop_words)
        relative_terms = set(termlist) - set(words_to_remove)
        return relative_terms

    @staticmethod
    def remove_stopword_frequency(termlist):
        """ removes the most frequent 100 terms in the entire document set """
        term_freqs = Counter(termlist)
        sorted_freqs = sorted(term_freqs.items(), key=operator.itemgetter(1))
        removed = sorted_freqs[:100]
        kept = sorted_freqs[100:]
        relative_terms = [i[0] for i in kept]
        return relative_terms, removed

    @staticmethod
    def term_frequency_vector(document):
        """ way to represent the term frequency for each of the terms in a document """
        term_frequency = Counter(document)
        doc_size = len(document)
        for term in term_frequency.keys():
            term_frequency[term] = term_frequency[term] / doc_size
        return term_frequency

    @staticmethod
    def term_count_vector(document):
        """ one way to represent terms by their frequency instead of sparse binary vector
         accepts a list of terms in the document """
        term_counter = Counter(document)
        return term_counter

    @staticmethod
    def parse_xml(file, query_doc=False):
        escaped_chars = {'\t"\t': "\t&quot;\t", '\t&\t': "\t&amp;\t", "\t'\t": "\t&apos;\t", "\t<\t": "\t&lt;\t",
                         "\t>\t": "\t&gt;\t"}
        newLines = file.split("\n")
        for char in escaped_chars:
            for line in range(len(newLines)):
                if char in newLines[line]:
                    newSub = escaped_chars[char]
                    test = re.sub(char, newSub, newLines[line])
                    newl = test.replace(char, newSub)
                    newLines[line] = newl

        lines = []
        line = 0
        if not query_doc:
            open_tag = "<DOC>"
            closing_tag = "</DOC>"
        else:
            open_tag = '<top lang="cs">'
            closing_tag = "</top>"

        if newLines[line] == open_tag:
            while newLines[line] != closing_tag:
                if newLines[line].split(" "):
                    lines.append(newLines[line])
                line += 1
        lines.append(closing_tag)
        formatted = "\n".join(lines)
        return formatted

    def parse_query(self, query_path, doc_terms, text_field):
        """
        function to parse the vectors for querying
        :param query_path: path to the document to query
        :param doc_terms: term forms to extract from the
        :param text_field:
        :return: dictionary that matches the search space
        """
        q_dict = {}
        with open(query_path, 'r') as q_document:
            try:
                root = etree.fromstring(q_document.read())
            except etree.ParseError:
                q_document.seek(0)
                clean_xml_str = self.parse_xml(q_document.read(), query_doc=True)
                root = etree.fromstring(clean_xml_str)
            text = root.find(text_field).text
            query_terms = self.get_doc_terms(text, doc_terms)
            q_dict['id'] = root.find('num').text.strip('\n')
            q_dict['terms'] = self.term_count_vector(query_terms)
        return q_dict


    def parse_document(self, file, doc_terms, text_field):
        """
        parses individual documents and returns a list of words and an id for each of the documents
        stop words, word forms and letter casings are also handles for individual documents
        :param file: file reader object to be parsed into the search space
        :param doc_terms: type of terms to search ie. forms, lemmas, etc.
        :param text_field: search terms to be searched ie. title or entire document
        :return: a tuple of document id, and terms
        """
        # check that the xml file is valid and create a list for escaped values not to pushed to query space
        file = file.decode("utf-8")
        try:
            root = etree.fromstring(file)
        except etree.ParseError:
            clean_xml_str = self.parse_xml(file)
            root = etree.fromstring(clean_xml_str)

        doc_id = root.find('DOCNO').text
        title = root.find('TITLE')
        if not title:
            title = ""
        else:
            title = title.text
        text = root.find('TEXT').text
        if text_field == title:
            all_text = title
        else:
            all_text = text + title
        # get all of the appropriate terms
        doc_terms = self.get_doc_terms(all_text, doc_terms)
        if self._lower_casing:
            for n in range(len(doc_terms)):
                doc_terms[n] = doc_terms[n].lower()
        # add all of the terms to the collection of documents which the term occurs in for further weights
        for term in set(doc_terms):
            if term not in self.inverse_doc_frequency:
                self.inverse_doc_frequency[term] = 1
            else:
                self.inverse_doc_frequency[term] += 1
            return doc_id, doc_terms


    def get_doc_collection(self):
        """ get the final format of the list of documents
            documents will be represented as a list of dictionaries that contain sparse vectors """
        total_tokens = 0
        for file in self._document_list:
            file = file + '.gz'
            try:
                with gzip.open(file) as doc:
                    doc_input = doc.read()
                    doc_dict = {}
                    try:
                        res = self.parse_document(doc_input, self._terms, self._lower_casing)
                        doc_id = res[0]
                        list_terms = res[1]
                        terms = self.term_count_vector(list_terms)
                        # append the document dictionary to the document indices with key elements of a document
                        doc_dict['Doc_ID'] = doc_id
                        doc_dict['terms'] = terms
                        doc_dict['length'] = len(list_terms)
                        total_tokens += doc_dict['length']
                        self.doc_indices.append(doc_dict)
                    except etree.ParseError:
                        # if file has multiple malfunctions go to the next file
                        continue
            except OSError:
                "File is not a gz file"
        self._avg_doc_len = total_tokens / len(self._document_list)
        return 0

    def get_collection_terms(self):
        """ get all of the terms of the  document collection"""
        buffer_terms = []
        for doc in self.doc_indices:
            terms = doc['terms'].keys()
            for term in terms:
                buffer_terms.append(term)
        collection_terms = sorted(set(buffer_terms))
        if self._stop_words is not "no":
            if self._stop_words is "frequency":
                stop_words_removal = self.remove_stopword_frequency(collection_terms)
                collection_terms = stop_words_removal[0]
                words_to_remove = stop_words_removal[1]
                # update the terms in the documents to match the collection terms
                self.remove_document_stopwords(words_to_remove)
            if self._stop_words is "lexicon":
                collection_terms = self.remove_stopwords_lexical(collection_terms)
                self.remove_document_stopwords(self._stop_words)
        term_index = 0
        for i in collection_terms:
            # set every term in the collection to an index starting from 0
            self.collection_term_indices[i] = term_index
            term_index += 1

        return 0

    def remove_document_stopwords(self, removal_list):
        """ removes words from each of the documents """
        for doc in range(len(self.doc_indices)):
            # remove terms from the set of terms to be removed
            removal_list = set(removal_list)
            self.doc_indices[doc] = {i: self.doc_indices[doc[i]] for i in doc if i not in removal_list}
        return 0

    def get_tf_idf(self, np_vector, term_dict):
        """ function to get tf-idf for a dictionary of terms and values"""
        ind_update = np.nonzero(np_vector)
        for x in ind_update:
            term_sum = np.sum(np_vector)
            N = len(self.doc_indices)
            for i in x:
                old_val = np_vector[i]
                tf = old_val / term_sum
                term = term_dict[i]
                if term not in self.inverse_doc_frequency:
                    # add one for smoothing
                    self.inverse_doc_frequency[term] = 1
                dt_f = self.inverse_doc_frequency[term]
                idf = math.log(N / dt_f)
                tf_idf = tf * idf
                np.put(np_vector, i, tf_idf)
        return np_vector

    @staticmethod
    def update_vector_weights(np_vector, function):
        """ helper function that updates the document vector with appropriate weights """
        ind_update = np.nonzero(np_vector)
        for x in ind_update:
            term_sum = np.sum(np_vector)
            for i in x:
                old_val = np_vector[i]
                tf = old_val / term_sum
                if function is "boolean":
                    np.put(np_vector, i, 1)
                elif function is "log":
                    new_val = 1 + math.log10(tf)
                    np.put(np_vector, i, new_val)
                else:
                    print("This weighting is not supported")
        return np_vector

    def transform_document_vector(self, document_dict, weight=None):
        """ this transforms a sparse vector to one that includes indices that
            correlate to terms in the collection set this representation of the
            document can either not be weighted and fall back to counts or use idf """
        # initialize the the vector so it is the size of the term list
        t_size_offset = len(self.collection_term_indices) - 1
        t_vector = np.zeros(len(self.collection_term_indices))
        term_tuple = {}
        # document dictionary is a frequency vector of word:frequency in document
        for term in document_dict['terms'].keys():
            # check if the term is in the term dictionary and get index
            if term in self.collection_term_indices:
                t_index = self.collection_term_indices[term]
                term_tuple[t_index] = term
                t_vector[t_index] = document_dict['terms'][term]
            else:
                # if term is not in the term dictionary add the index doc and term vector
                t_vector = np.append(t_vector, document_dict['terms'][term])
                t_size_offset += 1
                self.collection_term_indices[term] = t_size_offset
                term_tuple[self.collection_term_indices[term]] = term

        # get the tf-idf vector for normalization
        document_dict['tf-idf'] = self.get_tf_idf(t_vector, term_tuple)

        if weight is "boolean":
            t_vector = self.update_vector_weights(t_vector, "boolean")
        elif weight is "log":
            t_vector = self.update_vector_weights(t_vector, "log")
        elif weight is "tf-idf":
            t_vector = document_dict['tf-idf']
        document_dict['doc_vector'] = t_vector
        return

    @staticmethod
    def cos_normalize_vector(vector):
        """ takes a vector, and normalizes the vector based on the normalization method """
        ind = np.nonzero(vector)
        # get the indices of all the nonzero values
        magnitude = 0
        for i in ind:
            n = vector[i]**2
            magnitude += n
        magnitude = math.sqrt(magnitude)
        # loop through to update the values of the original vector
        norm_vector = vector / magnitude
        return norm_vector

    def update_document_vectors(self):
        """
        function to update the documents to have the appropriate output vectors
        :return:
        """
        for doc in self.doc_indices:
            doc_vector = self.transform_document_vector(doc, weight=self._term_weight)
            doc['doc_vector'] = doc_vector
        return

