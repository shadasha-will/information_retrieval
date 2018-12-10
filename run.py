#!usr/bin/env python
# Author: Shadasha Williams
# Description: A file that parses the input of the command line for the experiment runs

import argparse
from operator import itemgetter
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

import xml.etree.ElementTree as etree
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import Document_collection

"""parser = argparse.ArgumentParser(description='Parses input from command line for test runs')
parser.add_argument('-q', '--topics', required=True, type=str,
                        help='a file with the list of topic names')
parser.add_argument('-d','--docs', required=True, type=str,
                     help='a file with the list of document filenames')
parser.add_argument('-r', '--run',  required=True, type=str,
                    help='a label identifying the experiment being run')
parser.add_argument('-o','--output',  required=True, type=str, help='a name with the output file')
parser.add_argument('-f', '--frequency', required=False, type=str, help='describes a type of term frequency measure')
parser.add_argument('-n', '--normalization', required=False, type=str, help='a')
args = parser.parse_args()

file_list = args.docs
topic_docs = args.topics
run_name = args.run
results_file = args.output"""

file_list = 'documents.list'
topic_docs = 'test-topics.list'
results_file = 'run_0_train-res.dat'

class Run:
    def __init__(self, file_list, topic_list, experiment_name, out_file,  l_norm, term_weight, term_frequency_method=None):
        self._file_list = file_list
        self._topic_file = topic_list
        self._exp_name = experiment_name
        self._out_file = out_file
        self._l_norm = l_norm
        self._term_weight = term_weight
        if term_frequency_method is None:
            term_frequency_method = 'none'
        else:
            self._term_frequency_method = term_frequency_method

        self._doc_freq_method = term_frequency_method

    @staticmethod
    def cos_doc_similarity(q_vec, d_vec):
        """ function that creates a transformation of the query and does normalization and similarity of two vectors """
        q_idf_tf = q_vec['tf-idf']
        d_idf_tf = d_vec['tf-idf']
        # check if new terms were added to collection from query
        if q_idf_tf.shape[0] > d_idf_tf.shape[0]:
            new_terms = q_idf_tf.shape[0] - d_idf_tf.shape[0]
            new_cols = np.zeros(new_terms)
            d_idf_tf = np.append(d_idf_tf, new_cols)
        cos_dist = cosine_similarity([q_idf_tf], [d_idf_tf])
        cos_val = cos_dist[0][0]
        out_tuple = (cos_val, d_vec['Doc_ID'])
        return out_tuple

    def get_query_docs(self, folder_path, doc_collection, term_type):
        q_docs = open(folder_path, 'r').read().split('\n')
        for query in q_docs:
            with open(query, 'r') as q_file:
                xml_string = doc_collection.parse_xml(q_file)
                root = etree.fromstring(xml_string)
                # only read forms from the title
                q_id = root.find('num')
                title = root.find('TITLE')
                terms = doc_collection.get_doc_terms(title, term_type)
                # transform the query into a vector
                q_dict = doc_collection.term_count_vector(terms)
                q_vector = doc_collection.transform_document_vector(q_dict, self._doc_freq_method)
            return

    def run_SVM(self, exp_name, query_forms):
        """
        function to run the different experiments baseline is set as the default
        :param exp_name: name of the experiment being run
        :param query_forms: forms to query the document as ie. word forms, lemmas, stems
        :return: Null, but new file is created with the results
        """

        # baseline setup
        stop_words = False
        lower_casing = False
        search_field = "title"
        exp_id = "baseline"
        term_weight = None

        if exp_name != 'Run-0':
            if exp_name == 'Run-1':
                stop_words = True
            elif exp_name == 'Run-2':
                stop_words = True
                lower_casing = True

        exp = Document_collection.CollectionPipeline(terms=query_forms, stop_words=stop_words,
                                                     lower_casing=lower_casing, search_field=search_field,
                                                     term_weight=None,
                                                     document_list=self._file_list)
        exp.get_doc_collection()
        exp.get_collection_terms()
        exp.update_document_vectors()
        # get the names of the documents in the topic
        with open(self._out_file, 'w') as out_file:
            topic_list = open(self._topic_file, 'r').read().split('\n')
            for topic in topic_list:
                if topic:
                    topic_dict = exp.parse_query(topic, doc_terms=query_forms, text_field=search_field)
                    exp.transform_document_vector(topic_dict, weight=term_weight)
                    q_doc = topic_dict['doc_vector']
                    results = []
                    for doc in exp.doc_indices:
                        exp.transform_document_vector(doc, weight=term_weight)
                        collection_doc = doc['doc_vector']
                        if q_doc.shape[0] > collection_doc.shape[0]:
                            new_terms = q_doc.shape[0] - collection_doc.shape[0]
                            new_cols = np.zeros(new_terms)
                            collection_doc = np.append(collection_doc, new_cols)
                        sim = cosine_similarity([q_doc], [collection_doc])
                        sim_value = sim[0][0]
                        results.append((sim[0][0], doc['Doc_ID']))
                    # get the first 20 best results
                    top_20 = sorted(results, key=itemgetter(0))[:20]
                    # write results to the output file
                    rank = 0
                    for x in top_20:
                        line = [topic_dict['id'], "0", str(x[1]), str(rank), str(x[0]), exp_name, "\n"]
                        rank += 1
                        out_file.write('\t'.join(line))

this = Run(file_list=file_list, topic_list=topic_docs, experiment_name='Run-2', term_weight=None, out_file=results_file, l_norm=None)
this.run_SVM('Run-2', 'lemmas')










