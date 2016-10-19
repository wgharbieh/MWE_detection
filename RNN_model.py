
WORD2VEC_PATH = "/home/waseem/models/"
PATH = "/home/waseem/Downloads/dimsum-data-1.5/"

import tensorflow as tf
import gensim
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import ujson
import extract_mwes
import sys
import rnn_features
import skipthoughts
import operator
import error_analysis
import pickle
import string

WINDOW = 0
POS_WINDOW = 0
TAG_WINDOW = 0
MWE_WINDOW = 0
FEAT_WIN = 0
DIM = 100
embedding_window = 5
tag_features = 0
token_features = 1
lemma_features = 1
pos_features = 1
context = 1
in_mwe = 1
tag_distribution = 0
ngram = 2
rho = 50
cost_function = 'sce' #sce
#hidden = [(200,'tanh'),(50,'tanh'),(6,'soft')] # 400 relu, 100 tanh = 61.2%; 400 relu, 100 relu = <60%
hidden = [50,(6,'soft')]
#hidden = None
test = 0
max_iter = 100000
reg = 1e10
sentence_embedding = 0
learning_rate = 1e-2
gen_vec = 0
unk = 0 # Declaring unk = 1 to prevent zero vectors, can set this to 0 when POS_features = 1
remove_target = 0
normalize = 0
sample = 0
batch_size = 1000
uni_skip = 0
bi_skip = 0
Google_vecs = 0
con_win = 100
poshmm = 0
top_k_predictions = 3
char_win = 3
gram_win = 3
seed_val = 102
forward_pass = 1
backward_pass = 1
forward_reverse = 0
backward_reverse = 0
keep_prob = 1
clip_grad = 10


def parse_input(x):
    #print "x: " + str(x)
    if x.__contains__(','):
        k = x.split(',')
        y = ''.join(k)
        #print "y: " + str(y)
        t = ()
        for element in y:
            t += (int(element),)
        return t
    return int(x)


# WINDOW = parse_input(sys.argv[1])
# POS_WINDOW = parse_input(sys.argv[2])
# TAG_WINDOW = parse_input(sys.argv[3])
# MWE_WINDOW = parse_input(sys.argv[4])
# DIM = int(sys.argv[5])
# embedding_window = int(sys.argv[6]) #4
# tag_features = int(sys.argv[7])
# lemma_features = int(sys.argv[8])
# pos_features = int(sys.argv[9])
# context = int(sys.argv[10])
# in_mwe = int(sys.argv[11])
# ngram = int(sys.argv[12])
# rho = int(sys.argv[13])
# cost_function = sys.argv[14] #sce
# hidden = [(int(sys.argv[15]),'tanh'),(6,'soft')]
# test = int(sys.argv[16])
# max_iter = int(sys.argv[17])
# sentence_embedding = 0
# learning_rate = 1e-2
# #gen_vec = int(sys.argv[17])
# remove_target = int(sys.argv[18])
# normalize = int(sys.argv[19])
# con_win = int(sys.argv[20]) # 3 = 65.7%, 5 = 66.4%
# seed_val = int(sys.argv[21])

avg_tokens = 0
avg_lemmas = 0
cbow = 0
mincount = 15
WORD2VEC = "Word2Vec_" + str(DIM) + "d_" + str(embedding_window) + "window_" + str(mincount) + "mincount_5nsampling_"
if cbow == 1:
    WORD2VEC += "cbow"
else:
    WORD2VEC += "skipgram"
if unk == 1:
    WORD2VEC += '_UNK'
WORD2VEC += ".bin"
WORD2VEC2 = "wikiEmbeddingsStanfordTokenizedSkipGram" + str(embedding_window) + "-dimension" + str(DIM) + "vectors.bin"
TOKENS_PATH = PATH + 'tokens/'
LEMMAS_PATH = PATH + 'lemmas/'
unknown_token = 'UNK'
splits = 5
num = 1
repetitions = 1
 #400
resolution = 10
top_k = 10
cutoff = 100
chosen_words = [unknown_token]

#batch_size = 50000
debug_features = 0
debug_viterbi = 0

separate_lexicons = 0

if in_mwe == 1:
    tag_distribution = 0
sub_token_lemma = 0
 #10 gives 39.3% on k = 3, and win = 5
#20 gives 39.0% on k = 5 and win = 5, rho = 50 is a good parameter
pretrained = 0
pretrained_file = "weights_1000"
 #relu = 52.3% #tanh = 54.1% #sig = 53.5%
#hidden = [(205,'relu'),(35,'relu'),(6,'soft')] #tanh, tanh = 56.4%
#hidden = None
pos_tags = []
emission_arr = []
multiclass = []

start_tag = 'start'
end_tag = 'end'
tag_set = ['B', 'I', 'O', 'b', 'i', 'o']
#tag_set = ['B', 'I', 'O']

tf.set_random_seed(seed_val)

def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)


def orthogonal_initializer(shape, scale, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev= 0.1, seed=seed_val)
    return tf.Variable(initial)
    #Initialize with small weights


def bias_variable(shape):
    initial = tf.constant(0.1, shape= shape)
    return tf.Variable(initial)
    # Initialize with small positive weights


def tag_feat(tag,tag_set):
    feats = []
    for ts in tag_set:
        if ts == tag:
            feats.append(1)
        else:
            feats.append(0)
    return feats

def div(x,y):
    val = 0
    if x != 0 or y != 0:
        val = x*1.0/y
    return val

def write_predictions(classification, test_file):
    reference = open(test_file)
    new_file = open(test_file + '.pred','w')
    rline = reference.readline()
    i = 0
    sequence = []
    j = 1
    index1 = -1
    index2 = -1
    ii = 0
    while rline != '':
        if rline != '\n':
            k = rline.split('\t')
            current_tag = classification[i][ii]
            k[4] = current_tag
            if current_tag == 'B':
                index1 = j
            elif current_tag == 'b':
                index2 = j
            elif current_tag == 'I':
                k[5] = str(index1)
                index1 = j
            elif current_tag == 'i':
                k[5] = str(index2)
                index2 = j
            if current_tag != 'I' and current_tag != 'i':
                k[5] = '0'
            k[7] = ''
            ii += 1
            j += 1
            sequence.append(k)
        else:
            for element in sequence:
                new_file.write('\t'.join(element))
            new_file.write('\n')
            sequence = []
            j = 1
            ii = 0
            i += 1
        rline = reference.readline()
    reference.close()
    new_file.close()

def update_sequences(tag_sequences, p, pred, emission_arr, new_sequences, tag_set):
    for sequence in tag_sequences:
        if sequence[sequence.__len__() - 1] == p:
            for i in range(0, pred.__len__()):
                if pred.__len__() == tag_set.__len__():
                    current_tag = tag_set[i]
                    tag_prob = pred[i]
                else:
                    current_tag = pred[i]
                    tag_prob = 1
                tag_pair = (p,current_tag)
                if emission_arr.__contains__(tag_pair):
                    transition_probability = emission_arr[tag_pair]
                else:
                    transition_probability = 0
                temp = tag_sequences[sequence]*transition_probability*tag_prob
                temp_sequence = sequence + (current_tag,)
                visited = 0
                if new_sequences.__len__() > 0:
                    for new_seq in new_sequences:
                        if new_seq[new_seq.__len__() - 1] == current_tag:
                            if new_sequences[new_seq] < temp:
                                new_sequences.pop(new_seq)
                                new_sequences[temp_sequence] = temp
                            visited = 1
                            break
                if visited == 0:
                    new_sequences[temp_sequence] = temp

def display_features(feature):
    e = 0
    count = 1
    mwe_lexicon = (6*tag_distribution*int(WINDOW) + 18*separate_lexicons)*tag_distribution + (in_mwe + 9*separate_lexicons)*in_mwe
    threshold = feature.__len__() - mwe_lexicon*(int(WINDOW) + 1)
    while e < threshold:
        s = e
        e += DIM
        print "token " + str(count) + ":" + str(feature[s:e])
        if lemma_features == 1:
            s = e
            e += DIM
            print "lemma " + str(count) + ":" + str(feature[s:e])
        if pos_features == 1:
            s = e
            e += pos_tags[0].__len__()
            print "pos_tag " + str(count) + ":" + str(feature[s:e])
        if tag_features == 1:
            s = e
            e += tag_set.__len__()
            print "tag " + str(count) + ":" + str(feature[s:e])
        count += 1
    if avg_tokens == 1:
        s = e
        e += DIM
        print "avg_tokens " + str(feature[s:e])
    if avg_lemmas == 1:
        s = e
        e += DIM
        print "avg_lemmas " + str(feature[s:e])
    if separate_lexicons == 0:
        s = e
        if in_mwe == 1:
            e += 1
            print "in_mwe " + str(feature[s:e])
        if tag_distribution == 1:
            e += 6*int(WINDOW)
            print "tag_distribution " + str(feature[s:e])
    else:
        s = e
        if in_mwe == 1:
            e += 10
            print "in_mwe " + str(feature[s:e])
        if tag_distribution == 1:
            e += 20
            print "tag_distribution " + str(feature[s:e])

def Viterbi(sequence, tag_set, emission_prob):
    new_sequence = []
    tag_dict= []
    for i in range(0,sequence.__len__()):
        tag_dict.append({})
    bp = {}
    prev_tag = ()
    for i in range(0,ngram):
        prev_tag += (start_tag,)
    temp_dict = {}
    for tag in tag_set:
        current_tuple = prev_tag[1:] + (tag,)
        if debug_viterbi == 1:
            print "current_tuple: " + str(current_tuple)
        if emission_prob.__contains__(current_tuple):
            temp_dict[current_tuple] = emission_prob[current_tuple]*sequence[0][tag_set.index(tag)]
            bp[(0,current_tuple)] = prev_tag
    tag_dict[0] = temp_dict
    if debug_viterbi == 1:
        print "sequence[0] = " + str(sequence[0])
        print tag_dict
    for i in range(1,sequence.__len__()): # recheck
        count = 0
        temp_dict = {}
        if debug_viterbi == 1:
            print "sequence[" + str(i) + "] = " + str(sequence[i])
        for prev_tag in tag_dict[i-1]:
            #prev_tag = tuples[tuples.__len__() - 1]
            if debug_viterbi == 1:
                print "For tuple " + str(prev_tag)
            for tag in tag_set:
                current_tuple = prev_tag[1:] + (tag,)
                if emission_prob.__contains__(current_tuple):
                    count += 1
                    temp = tag_dict[i-1][prev_tag]*sequence[i][tag_set.index(tag)]*emission_prob[current_tuple]
                    if temp_dict.has_key(current_tuple):
                        if temp > temp_dict[current_tuple]:
                            temp_dict[current_tuple] = temp
                            bp[(i,current_tuple)] = prev_tag
                    else:
                        temp_dict[current_tuple] = temp
                        bp[(i,current_tuple)] = prev_tag
                    if debug_viterbi == 1:
                        print "tag_dict[(" + str(i) + "," + str(current_tuple) + ")] = " + str(temp_dict[current_tuple])
                        print "bp[(" + str(i) + "," + str(current_tuple) + ")] = " + str(bp[(i,current_tuple)])
        tag_dict[i] = temp_dict
        if count == 0:
            print "Too sparse!!"
    tag = end_tag
    i = sequence.__len__()
    max_tuple = {}
    count = 0
    temp_dict = {}
    for prev_tag in tag_dict[i-1]:
        #prev_tag = tuples[tuples.__len__() - 1]
        tag = end_tag
        current_tuple = prev_tag[1:] + (tag,)
        if emission_prob.__contains__(current_tuple):
            count += 1
            temp = tag_dict[i-1][prev_tag]*emission_prob[current_tuple]
            if temp_dict.has_key(current_tuple):
                if temp > temp_dict[current_tuple]:
                    temp_dict[current_tuple] = temp
                    bp[(i,current_tuple)] = prev_tag
                    max_tuple[current_tuple] = temp
            else:
                temp_dict[current_tuple] = temp
                bp[(i,current_tuple)] = prev_tag
                max_tuple[current_tuple] = temp
            if debug_viterbi == 1:
                print "tag_dict[(" + str(i) + "," + str(current_tuple) + ")] = " + str(temp_dict[current_tuple])
                print "bp[(" + str(i) + "," + str(current_tuple) + ")] = " + str(bp[(i,current_tuple)])
        #tag_dict[i] = temp_dict
    max_value = -1
    best_tuple = ()
    #print "max_tuple: " + str(max_tuple)
    for tags in max_tuple:
        if max_tuple[tags] > max_value:
            max_value = max_tuple[tags]
            best_tuple = tags
    if count == 0:
        print "Too sparse!!"
    try:
        tag = best_tuple[best_tuple.__len__() - 1]
    except:
        print "best_tuple: " + str(best_tuple)
    i = sequence.__len__()
    #print "best_tuple: " + str(best_tuple)
    while tag != start_tag:
        #print "i = " + str(i) + ", max_tuple = " + str(max_tuple)
        #print bp
        best_tuple = bp[(i,best_tuple)]
        tag = best_tuple[best_tuple.__len__() - 1]
        #print "tag: " + str(tag)
        new_sequence.append(tag)
        i -= 1
    new_sequence.reverse()
    if new_sequence.__len__() != sequence.__len__() + 1:
        print "Wrong sequence"
        print new_sequence
        print sequence
    if debug_viterbi == 1:
        print new_sequence[1:]
        tag_dict[sequence.__len__()] = 1
    return new_sequence[1:]


def Viterbi_poshmm(sequence, tag_set, pos_tags, emission_prob):
    new_sequence = []
    tag_dict= []
    for i in range(0,sequence.__len__()):
        tag_dict.append({})
    bp = {}
    prev_tag = ()
    for i in range(0,ngram):
        tag_pos = (start_tag,)
        prev_tag += (tag_pos,)
    temp_dict = {}
    pair_set = []
    for tag in tag_set:
        for pos in pos_tags:
            pair_set.append((tag,pos))
    for tag in pair_set:
        current_tuple = prev_tag[1:] + (tag,)
        if debug_viterbi == 1:
            print "current_tuple: " + str(current_tuple)
        if emission_prob.__contains__(current_tuple):
            temp_dict[current_tuple] = emission_prob[current_tuple]*sequence[0][tag_set.index(tag[0])]
            bp[(0,current_tuple)] = prev_tag
    tag_dict[0] = temp_dict
    if debug_viterbi == 1:
        print "sequence[0] = " + str(sequence[0])
        print tag_dict
    for i in range(1,sequence.__len__()): # recheck
        count = 0
        temp_dict = {}
        if debug_viterbi == 1:
            print "sequence[" + str(i) + "] = " + str(sequence[i])
        for prev_tag in tag_dict[i-1]:
            #prev_tag = tuples[tuples.__len__() - 1]
            if debug_viterbi == 1:
                print "For tuple " + str(prev_tag)
            for tag in pair_set:
                current_tuple = prev_tag[1:] + (tag,)
                if emission_prob.__contains__(current_tuple):
                    count += 1
                    temp = tag_dict[i-1][prev_tag]*sequence[i][tag_set.index(tag[0])]*emission_prob[current_tuple]
                    if temp_dict.has_key(current_tuple):
                        if temp > temp_dict[current_tuple]:
                            temp_dict[current_tuple] = temp
                            bp[(i,current_tuple)] = prev_tag
                    else:
                        temp_dict[current_tuple] = temp
                        bp[(i,current_tuple)] = prev_tag
                    if debug_viterbi == 1:
                        print "tag_dict[(" + str(i) + "," + str(current_tuple) + ")] = " + str(temp_dict[current_tuple])
                        print "bp[(" + str(i) + "," + str(current_tuple) + ")] = " + str(bp[(i,current_tuple)])
        tag_dict[i] = temp_dict
        if count == 0:
            print "Too sparse!!"
    tag = end_tag
    i = sequence.__len__()
    max_tuple = {}
    count = 0
    temp_dict = {}
    for prev_tag in tag_dict[i-1]:
        #prev_tag = tuples[tuples.__len__() - 1]
        tag = (end_tag,)
        current_tuple = prev_tag[1:] + (tag,)
        if emission_prob.__contains__(current_tuple):
            count += 1
            temp = tag_dict[i-1][prev_tag]*emission_prob[current_tuple]
            if temp_dict.has_key(current_tuple):
                if temp > temp_dict[current_tuple]:
                    temp_dict[current_tuple] = temp
                    bp[(i,current_tuple)] = prev_tag
                    max_tuple[current_tuple] = temp
            else:
                temp_dict[current_tuple] = temp
                bp[(i,current_tuple)] = prev_tag
                max_tuple[current_tuple] = temp
            if debug_viterbi == 1:
                print "tag_dict[(" + str(i) + "," + str(current_tuple) + ")] = " + str(temp_dict[current_tuple])
                print "bp[(" + str(i) + "," + str(current_tuple) + ")] = " + str(bp[(i,current_tuple)])
        #tag_dict[i] = temp_dict
    max_value = 0
    best_tuple = ()
    for tags in max_tuple:
        if max_tuple[tags] > max_value:
            max_value = max_tuple[tags]
            best_tuple = tags
    if count == 0:
        print "Too sparse!!"
    if debug_viterbi == 1:
        print "best_tuple: " + str(best_tuple)
    i = best_tuple.__len__() - 2
    while tag != start_tag and i >= 0:
        tag = best_tuple[i]
        new_sequence.append(best_tuple[i][0])
        i -= 1
    tag = best_tuple[i+1]
    i = sequence.__len__()
    #print 'new_sequence: ' + str(new_sequence)
    while tag != (start_tag,):
        #print "i = " + str(i) + ", max_tuple = " + str(max_tuple)
        #print bp
        best_tuple = bp[(i,best_tuple)]
        tag = best_tuple[0]
        new_sequence.append(tag[0])
        i -= 1
    #print 'new_sequence: ' + str(new_sequence)
    new_sequence.reverse()
    if new_sequence.__len__() != sequence.__len__() + 1:
        print "Wrong sequence"
        print new_sequence
        print sequence
    if debug_viterbi == 1:
        print new_sequence[1:]
        tag_dict[sequence.__len__()] = 1
    return new_sequence[1:]

def get_max_predictions(new_sequences):
    max_arr = []
    for max_seq in new_sequences:
        for i in range(1, max_seq.__len__() - 1):
            max_arr.append(max_seq[i])
    return max_arr


def get_graph(sequence):
    graph = []
    source1 = -1
    sink1 = -1
    source2 = -1
    sink2 = -1
    for i in range(0,sequence.__len__()):
        if sequence[i] == 'B':
            source1 = i
        elif sequence[i] == 'I':
            sink1 = i
            graph.append((source1,sink1))
            source1 = i
        if sequence[i] == 'b':
            source2 = i
        elif sequence[i] == 'i':
            sink2 = i
            graph.append((source2,sink2))
            source2 = i
    #print sequence
    #print graph
    return graph


def equate_graphs(gold, prediction):
    tp = 0
    fp = 0
    inter = -1
    for pair in prediction:
        found = 0
        for pair2 in gold:
            if pair == pair2:
                tp += 1
                found = 1
                break
            elif pair[0] == pair2[0]:
                found = 2
                inter = pair2[1]
            elif found == 2 and pair[1] == pair2[1]:
                found = 1
                break
            elif found == 2 and inter == pair2[0]:
                inter = pair2[1]
        if found != 1:
            fp +=1
    #print gold
    #print prediction
    #print "tp: " + str(tp) + ", fp: " + str(fp)
    return tp, fp


def confusion_matrix(gold, prediction):
    #gold_tags = get_tags(gold, tag_set)
    #print "gold_tags: " + str(gold)
    #prediction_tags = get_tags(prediction, tag_set)
    #print "prediction_tags: " + str(prediction)
    gold_sequence = get_graph(gold)
    pred_sequence = get_graph(prediction)
    stats = error_analysis.confusion_stats(gold, prediction, tag_set)
    tp1, fp = equate_graphs(gold_sequence, pred_sequence)
    tp2, fn = equate_graphs(pred_sequence, gold_sequence)
    return tp1, tp2, fp, fn, stats


def get_fscore(precision,recall):
    fscore = 0
    if precision != 0 and recall != 0:
        fscore = 2*precision*recall/(precision + recall)
    return fscore


def get_stats(classification):
    dict = {}
    for val1 in classification:
        for val in val1:
            index = tag_set.index(val)
            if dict.__contains__(tag_set[index]):
                dict[tag_set[index]] += 1
            else:
                dict[tag_set[index]] = 1
    return dict


def get_pred(test_file,validation_features,sess,forward_x,backward_x,W,b,y,model = None,final = 0):
    predictions = []
    gold_sequence = []
    gold_tags = []
    sentences = 0
    inside = 0
    keep_track = open(test_file)
    line = keep_track.readline()
    analyze = 0
    if final == 1:
        analyze = open(test_file + ".FA", 'w')
    j = 0
    tp1 = 0
    tp2 = 0
    fp = 0
    fn = 0
    count = 0
    global_matrix = np.zeros([tag_set.__len__(), tag_set.__len__()])
    sequence = []
    factor = DIM
    if tag_features == 1:
        factor += tag_set.__len__()
    if lemma_features == 1:
        factor += DIM
    if pos_features == 1:
        factor += pos_tags[0].__len__()
    line_by_line = []
    token_arr = []
    lemma_arr = []
    pos_arr = []
    token_sent = []
    lemma_sent = []
    pos_sent = []
    correct = 0
    total = 0
    prev_tags = (start_tag,)
    tag_sequences = {prev_tags: 1}
    while line != '':
        if line != '\n':
            k = line.split('\t')
            token = k[1]
            modified_token = rnn_features.preprocess(token)
            if model.__contains__(modified_token):
                model_token = modified_token
            elif model.__contains__(modified_token.lower()):
                model_token = modified_token.lower()
            else:
                model_token = unknown_token
            token_sent.append(model_token)
            lemma = k[2]
            modified_lemma = rnn_features.preprocess(lemma)
            if model.__contains__(modified_lemma):
                model_lemma = modified_lemma
            elif model.__contains__(modified_lemma.lower()):
                model_lemma = modified_lemma.lower()
            else:
                model_lemma = unknown_token
            lemma_sent.append(model_lemma)
            pos = k[3]
            pos_sent.append(pos)
            tag = k[4]
            line_arr = [k[0], k[1], model_token, k[2], model_lemma, k[3], k[4]]
            line_by_line.append(line_arr)
            new_sequences = {}
            forward_batch = 0
            backward_batch = 0
            if forward_pass == 1:
                forward_batch = [validation_features[0][j]]
            if backward_pass == 1:
                backward_batch = [validation_features[1][j]]
            if tag_features == 1:
                for p in prev_tags:
                    tag_vector = tag_feat(p, tag_set)
                    validation_features[j][0:6] = tag_vector
                    pred = run_model(sess, y, forward_x, forward_batch, backward_x, backward_batch, forward_pass, backward_pass)
                    update_sequences(tag_sequences, p, pred[0], emission_arr[0], new_sequences, tag_set)
                tag_sequences = new_sequences
                prev_tags = tag_set
            else:
                pred = run_model(sess, y, forward_x, forward_batch, backward_x, backward_batch, forward_pass, backward_pass)
            cl = np.zeros(pred[0].__len__())
            #cl,inside = amend_pred(pred[0],inside)
            #print "Classification: " + str(cl)
            #predictions.append(cl.index(1))
            #print "The prediction is " + str(cl.index(1))
            sequence.append(pred[0])
            pred_tags = np.argsort(pred[0])[::-1]
            gold_sequence.append(tag)
            for pred_index in range(0,top_k_predictions):
                if tag_set[pred_tags[pred_index]] == tag:
                    correct += 1
            total += 1
            if debug_features == 1:
                display_features(validation_features[j])
                count += 1
            j += 1
            if count == 30:
                print "check this out" + 5
            # print  "validation features shape: " + str(np.shape(validation_features))
        else:
            sentences += 1
            inside = 0
            if tag_features == 0:
                if poshmm == 0:
                    max_predictions = Viterbi(sequence, tag_set, emission_arr[0])
                else:
                    max_predictions = Viterbi_poshmm(sequence, tag_set, pos_tags[0], emission_arr[0])
            else:
                new_sequences = {}
                for p in tag_set:
                    update_sequences(tag_sequences, p, [end_tag], emission_arr[0], new_sequences, tag_set)
                max_predictions = get_max_predictions(new_sequences)
            #print "max_predictions: " + str(max_predictions)
            predictions.append(max_predictions)
            #print "gold_sequence: " + str(gold_sequence)
            #print "sequence: " + str(sequence)
            temp_tp1, temp_tp2, temp_fp, temp_fn, temp_stats = confusion_matrix(gold_sequence, max_predictions)
            pr = div(temp_tp1, temp_tp1 + temp_fp)
            assert pr <= 1
            re = div(temp_tp2, temp_tp2 + temp_fn)
            assert re <= 1
            if final == 1 and pr != 1 and re != 1 and (temp_tp1 + temp_fp + temp_tp2 + temp_fn) != 0:
                for iteration in range(0,max_predictions.__len__()):
                    line_by_line[iteration].append(max_predictions[iteration])
                    analyze.write('\t'.join(line_by_line[iteration]))
                    analyze.write(('\n'))
                analyze.write('\n')
            tp1 += temp_tp1
            tp2 += temp_tp2
            fp += temp_fp
            fn += temp_fn
            global_matrix = np.add(global_matrix,temp_stats)
            #print predictions
            sequence = []
            gold_tags.append(gold_sequence)
            token_arr.append(token_sent)
            lemma_arr.append(lemma_sent)
            pos_arr.append(pos_sent)
            gold_sequence = []
            token_sent = []
            lemma_sent = []
            pos_sent = []
            line_by_line = []
            prev_tags = (start_tag,)
            tag_sequences = {prev_tags: 1}
        line = keep_track.readline()
    if final == 1:
        analyze.close()
    keep_track.close()
    precision = div(tp1,tp1+fp)
    recall = div(tp2,tp2 + fn)
    fscore = get_fscore(precision,recall)
    acc_val = fscore
    train_criteria = get_stats(predictions)

    # print "\nTop " + str(top_k) + " mistakes:"
    # error_analysis.error_stats([token_arr, lemma_arr, pos_arr],[top_k,top_k,top_k],gold_tags, predictions)
    # print "\nTop " + str(top_k) + " mistakes that occurred at least " + str(cutoff) + " times:"
    # error_analysis.stats_by_percentage([token_arr, lemma_arr, pos_arr],[top_k,top_k,top_k],[cutoff, cutoff, cutoff],gold_tags, predictions)
    # word_features = ['cap','anycap','upper','alpha','num','nonchar','http']
    # print "\nTop " + str(top_k) + " word/lemma mistakes:"
    # error_analysis.word_stats([token_arr, lemma_arr], [top_k, top_k], word_features, gold_tags, predictions)
    # print "\nTop " + str(top_k) + " word/lemma mistakes that occurred at least " + str(cutoff) + " times:"
    # error_analysis.word_stats_by_percentage([token_arr, lemma_arr],[top_k,top_k],[cutoff,cutoff],word_features,gold_tags, predictions)
    # print "\nTop " + str(top_k) + " pair mistakes:"
    # error_analysis.analyze_pair(token_arr, pos_arr, top_k, gold_tags, predictions)
    # print "\nTop " + str(top_k) + " pair mistakes that occurred at least " + str(cutoff) + " times:"
    # error_analysis.analyze_pair_by_percentage(token_arr, pos_arr, top_k, cutoff, gold_tags, predictions)
    # for choose in chosen_words:
    #     print "Deeper analysis of " + str(choose)
    #     error_analysis.analyze_branches(token_arr, pos_arr, choose, gold_tags, predictions)

    # print "\nTop " + str(top_k) + " beginning character mistakes:"
    # error_analysis.analyze_characters([token_arr, lemma_arr], [top_k, top_k], char_win, gold_tags, predictions)
    # error_analysis.analyze_characters_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], char_win, gold_tags, predictions)
    # print "\nTop " + str(top_k) + " ending character mistakes:"
    # error_analysis.analyze_characters([token_arr, lemma_arr], [top_k, top_k], -1*char_win, gold_tags, predictions)
    # error_analysis.analyze_characters_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], -1*char_win, gold_tags, predictions)
    #
    # print "\nTop " + str(top_k) + " beginning " + str(gram_win) + "-gram mistakes:"
    # error_analysis.analyze_ngrams([token_arr, lemma_arr], [top_k, top_k], -1*gram_win, gold_tags, predictions)
    # error_analysis.analyze_ngrams_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], -1*gram_win, gold_tags, predictions)
    # print "\nTop " + str(top_k) + " ending " + str(gram_win) + "-gram mistakes:"
    # error_analysis.analyze_ngrams([token_arr, lemma_arr], [top_k, top_k], gram_win, gold_tags, predictions)
    # error_analysis.analyze_ngrams_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], gram_win, gold_tags, predictions)
    #
    # print "\nTop " + str(top_k) + " beginning " + str(gram_win) + " window mistakes:"
    # error_analysis.analyze_window([token_arr, lemma_arr], [top_k, top_k], -1*gram_win, gold_tags, predictions)
    # error_analysis.analyze_window_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], -1*gram_win, gold_tags, predictions)
    # print "\nTop " + str(top_k) + " ending " + str(gram_win) + " window mistakes:"
    # error_analysis.analyze_window([token_arr, lemma_arr], [top_k, top_k], gram_win, gold_tags, predictions)
    # error_analysis.analyze_window_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], gram_win, gold_tags, predictions)
    #
    # print "\nTop " + str(top_k) + " " + str(gram_win) + " context window mistakes:"
    # error_analysis.analyze_context([token_arr, lemma_arr], [top_k, top_k], gram_win, gold_tags, predictions)
    # error_analysis.analyze_context_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], gram_win, gold_tags, predictions)
    #
    # print "\nTop " + str(top_k) + " " + str(gram_win) + " group mistakes:"
    # error_analysis.analyze_group([token_arr, lemma_arr], [top_k, top_k], gram_win, gold_tags, predictions)
    # error_analysis.analyze_group_by_precentage([token_arr, lemma_arr], [top_k, top_k], [cutoff, cutoff], gram_win, gold_tags, predictions)
    #
    # if predictions.__len__() != gold_tags.__len__():
    #     print "Predictions and labels are not equal!!!"
    # print "Accuracy = " + str(correct*100.0/total)
    # print train_criteria
    #if train_criteria.has_key('B'):
    #    if train_criteria['B'] >= 120:
    #        break
    #print acc_train
    return tp1, fp, tp2, fn, acc_val, predictions, global_matrix

def summary_statement(ii, acc_train, acc_val, tp1, fp, tp2, fn, matrix):
    print str(ii) + ": " + " training: " + str(acc_train) + " validation: " + str(acc_val)
    print str(ii) + ": " + " precision: " + str(tp1) + "/" + str(tp1+fp) + " = " + str(div(tp1,tp1+fp)) +\
    "\t" + " recall: " + str(tp2) + "/" + str(tp2+fn) + " = " + str(div(tp2,tp2+fn))
    print '\t\t' + '\t\t'.join(tag_set)
    np.set_printoptions(suppress=True)
    #np.set_printoptions(precision=3)
    print matrix


def get_maximum_length(features):
    max_len = 0
    for inputs in features:
        if inputs.__len__() > max_len:
            max_len = inputs.__len__()
    return max_len


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def normalize_features(sequences, seq_len, some_pass):
    if some_pass == 1:
        single_seq = sequences[0]
        feat_len = single_seq[0].__len__()
        for sequence in sequences:
            len = sequence.__len__()
            for i in range(len, seq_len):
                sequence.append(np.zeros(feat_len))

def run_model(sess, parameter, forward_x, forward_batch_xs, backward_x, backward_batch_xs, forward_pass, backward_pass, y_ = None, batch_ys = None):
    if y_ == None:
        if forward_pass == 1 and backward_pass == 1:
            return sess.run(parameter, feed_dict={forward_x: forward_batch_xs, backward_x: backward_batch_xs})
        elif forward_pass == 1:
            return sess.run(parameter, feed_dict={forward_x: forward_batch_xs})
        elif backward_pass == 1:
            return sess.run(parameter, feed_dict={backward_x: backward_batch_xs})
    else:
        if forward_pass == 1 and backward_pass == 1:
            return sess.run(parameter, feed_dict={forward_x: forward_batch_xs, backward_x: backward_batch_xs, y_: batch_ys})
        elif forward_pass == 1:
            return sess.run(parameter, feed_dict={forward_x: forward_batch_xs, y_: batch_ys})
        elif backward_pass == 1:
            return sess.run(parameter, feed_dict={backward_x: backward_batch_xs, y_: batch_ys})
    print "Returning ZERO! forward_pass = " + str(forward_pass) + ", backward_pass = " + str(backward_pass) + ", y_ = " + str(y_)
    return 0

def train_RNN(training_features,training_labels, validation_features, validation_labels, classes, cost_fun, alpha, stop, predict, rho, batch_size, test_file, model = None, hidden = None):
    acc_train = 0
    acc_val = 0
    forward_features = training_features[0]
    backward_features = training_features[1]
    input_size = 0
    if forward_pass == 1:
        input_size = forward_features[0][0].__len__()
    elif backward_pass == 1:
        input_size = backward_features[0][0].__len__()
    print "input_size: " + str(input_size)
    forward_seq_len = get_maximum_length(forward_features)
    backward_seq_len = get_maximum_length(backward_features)
    forward_validation = validation_features[0]
    backward_validation = validation_features[1]
    forward_validation_seq_len = get_maximum_length(forward_validation)
    backward_validation_seq_len = get_maximum_length(backward_validation)
    if forward_validation_seq_len > forward_seq_len:
        forward_seq_len = forward_validation_seq_len
    if backward_validation_seq_len > backward_seq_len:
        backward_seq_len = backward_validation_seq_len

    normalize_features(forward_features, forward_seq_len, forward_pass)
    normalize_features(backward_features, backward_seq_len, backward_features)
    normalize_features(forward_validation, forward_seq_len, forward_pass)
    normalize_features(backward_validation, backward_seq_len, backward_pass)

    print "forward_features: " + str(np.shape(forward_features))
    print "backward_features: " + str(np.shape(backward_features))
    print "forward_validation: " + str(np.shape(forward_validation))
    print "backward_validation: " + str(np.shape(backward_validation))

    forward_x = tf.placeholder(tf.float32, [None, forward_seq_len, input_size])
    backward_x = tf.placeholder(tf.float32, [None, backward_seq_len, input_size])
    y_ = tf.placeholder(tf.float32, [None, classes])
    num_hidden = hidden[0]
    forward_last = 0
    backward_last = 0
    with tf.variable_scope('forward'):
        forward_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple= True)
    with tf.variable_scope('backward'):
        backward_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    if keep_prob < 1:
        forward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(forward_lstm_cell, output_keep_prob=keep_prob) #Try the longer memory variant
        backward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(backward_lstm_cell, output_keep_prob=keep_prob)  # Try the longer memory variant
    if forward_pass == 1:
        print "In forward pass"
        with tf.variable_scope('forward'):
            f1_outputs, f_states = tf.nn.dynamic_rnn(cell=forward_lstm_cell, inputs=forward_x, sequence_length= length(forward_x), dtype=tf.float32)
            print "computed forward outputs and states"
            #f2_outputs = tf.transpose(f1_outputs, [1,0,2])
            #forward_last = tf.gather(f2_outputs, tf.to_int32(tf.shape(f2_outputs)[2]) - 1)
            forward_last = last_relevant(f1_outputs, length(forward_x))
            last_outputs = forward_last
        print "outputs: " + str(last_outputs.get_shape())
    if backward_pass == 1:
        print "In backward pass"
        with tf.variable_scope('backward'):
            b1_outputs, b_states = tf.nn.dynamic_rnn(cell=backward_lstm_cell, inputs=backward_x, sequence_length= length(backward_x) , dtype=tf.float32)
            print "computed backward outputs and states"
            #b2_outputs = tf.transpose(b1_outputs, [1,0,2])
            #backward_last = tf.gather(b2_outputs, tf.to_int32(tf.shape(b2_outputs)[2]) - 1)
            backward_last = last_relevant(b1_outputs, length(backward_x))
            last_outputs = backward_last
        print "outputs: " + str(last_outputs.get_shape())
    if forward_pass == 1 and backward_pass == 1:
        print "Both forward and backward"
        last_outputs = tf.concat(1, [forward_last, backward_last])
        print "Concatenated forward and backward"
        print "outputs: " + str(last_outputs.get_shape())
    #x_input = [x]
    #Holds variables, 'None' indicates unknown length
    W = []
    b = []
    y = 0
    if hidden.__len__() >= 2:
        l1 = 0
        if forward_pass == 1:
            l1 += num_hidden
        if backward_pass == 1:
            l1 += num_hidden
        #h_fc1_drop = tf.nn.dropout(y1, keep_prob)
        l2 = hidden[1][0]
        W2 = weight_variable([l1,l2])
        #This is how a variable is declared, the model parameters should be declared as variables
        b2 = bias_variable([l2])
        activation = hidden[1][1]
        print "last: " + str(last_outputs.get_shape())
        print "W2: " + str(W2.get_shape())
        intermediate = tf.matmul(last_outputs,W2) + b2
        if activation == 'tanh':
            y2 = tf.nn.tanh(intermediate)
        elif activation == 'sig':
            y2 = tf.nn.sigmoid(intermediate)
        elif activation == 'relu':
            y2 = tf.nn.relu(intermediate)
        elif activation == 'soft':
            y2 = tf.nn.softmax(intermediate)
        y = y2
        if hidden.__len__() == 3:
            l3 = hidden[2][0]
            W3 = weight_variable([l2,l3])
            #This is how a variable is declared, the model parameters should be declared as variables
            b3 = bias_variable([l3])
            activation = hidden[2][1]
            intermediate = tf.matmul(y2,W3) + b3
            if activation == 'tanh':
                y3 = tf.nn.tanh(intermediate)
            elif activation == 'sig':
                y3 = tf.nn.sigmoid(intermediate)
            elif activation == 'relu':
                y3 = tf.nn.relu(intermediate)
            elif activation == 'soft':
                y3 = tf.nn.softmax(intermediate)
            y = y3
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    #print validation_labels[:100]
    if cost_fun == 'ce':
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    elif cost_fun == 's':
        #print [validation_labels[:2]]
        #print str(sess.run(y,feed_dict={x:validation_features[:2]}))
        part1 = tf.slice(y_,[0,0],[-1,1])
        #eval1 = sess.run(part1,feed_dict={y_: validation_labels[:2]})
        #print "part1: " + str(eval1)
        part2 = tf.slice(y,[0,2],[-1,1])
        #eval2 = sess.run(part2,feed_dict={x: validation_features[:2]})
        #print "part2: " + str(eval2)
        AND = tf.reshape(tf.mul(part1,part2), [-1])
        #print "AND: " + str(sess.run(AND, feed_dict={part1: eval1, part2: eval2}))
        #cast = tf.cast(AND, tf.float32)
        #cast_num = sess.run(cast, feed_dict={part1: eval1, part2: eval2})
        #print "cast: " + str(cast_num)
        #p1 = tf.arg_max(y_,1)
        #amax = sess.run(p1,feed_dict={y_: validation_labels[:2]})
        #print "arg_max: " + str(amax)
        eq = tf.reduce_sum(tf.abs(y - y_),reduction_indices=[1])
        #c = tf.cast(eq,tf.float32)
        #eq_num = sess.run(eq, feed_dict={x: validation_features[0:2], y_: validation_labels[:2]})
        #print "y and y_ are equal? " + str(eq_num)
        ADD = tf.add(eq,rho*AND)
        #ADD_num = sess.run(ADD, feed_dict={x: validation_features[:2], y_: validation_labels[:2]})
        #print "Addition result = " + str(ADD_num)
        cross_entropy = tf.reduce_mean(ADD)
        #experiment = sess.run(cross_entropy, feed_dict={x: validation_features[:2], y_: validation_labels[:2]})
        #print "CE value: " + str(experiment)
    elif cost_fun == 'sce':
        #start = 15
        #end = 20
        #print [validation_labels[start:end]]
        #print str(sess.run(y,feed_dict={x:validation_features[start:end]}))
        part1a = tf.slice(y_,[0,0],[-1,1])
        #print "part1a: " + str(sess.run(part1a, feed_dict={y_:validation_labels[start:end]}))
        #part1b = tf.slice(y_, [0, 3], [-1, 3])
        #print "part1b: " + str(sess.run(part1b, feed_dict={y_: validation_labels[start:end]}))
        part2 = tf.slice(y,[0,2],[-1,1])
        #print "part2: " + str(sess.run(part2, feed_dict={x: validation_features[start:end]}))
        AND1 = tf.reduce_sum(tf.mul(part1a,part2), reduction_indices=1)
        #print "AND1: " + str(sess.run(AND1, feed_dict={x: validation_features[start:end], y_: validation_labels[start:end]}))
        #AND2 = tf.reduce_sum(tf.mul(part1b, part2), reduction_indices=1)
        #print "AND2: " + str(sess.run(AND2, feed_dict={x: validation_features[start:end], y_: validation_labels[start:end]}))
        eq = tf.nn.softmax_cross_entropy_with_logits(intermediate,y_)
        #print "eq: " + str(sess.run(eq, feed_dict={x: validation_features[start:end], y_: validation_labels[start:end]}))
        #ADD = tf.add(eq,rho*(AND1 + AND2))
        ADD = tf.add(eq, rho * (AND1))
        #print "ADD: " + str(sess.run(ADD, feed_dict={x: validation_features[start:end], y_: validation_labels[start:end]}))
        cross_entropy = tf.reduce_mean(ADD)
        #print "cross_entropy: " + str(sess.run(cross_entropy, feed_dict={x: validation_features[start:end], y_: validation_labels[start:end]}))
    elif cost_fun == 'sce2':
        print [validation_labels[:2]]
        #print str(sess.run(y,feed_dict={x:validation_features[:2]}))
        part1 = tf.slice(y_,[0,0],[-1,1])
        part2 = tf.slice(y,[0,2],[-1,1])
        AND = tf.reshape(tf.mul(part1,part2), [-1])
        #eq1 = -tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1])
        #eq2 = -tf.reduce_sum((1-y_) * tf.log(1-y), reduction_indices=[1])
        eq1 = tf.nn.softmax_cross_entropy_with_logits(intermediate,y_)
        eq2 = tf.nn.softmax_cross_entropy_with_logits(1-intermediate, 1-y_)
        eq = tf.add(eq1, eq2)
        #eq_num = sess.run(eq, feed_dict={x: validation_features[0:2], y_: validation_labels[:2]})
        #print "y and y_ are equal? " + str(eq_num)
        ADD = tf.add(eq,rho*AND)
        cross_entropy = tf.reduce_mean(ADD)
    elif cost_fun == 'cus':
        part1 = tf.slice(y_,[0,1],[-1,1])
        part2 = tf.slice(y,[0,1],[-1,1])
        AND1 = -tf.reshape(tf.add(tf.mul(part1,tf.log(part2)),tf.mul(1-part1,tf.log(1-part2))), [-1])
        p1 = tf.slice(y_,[0,4],[-1,1])
        p2 = tf.slice(y,[0,4],[-1,1])
        AND2 = -tf.reshape(tf.add(tf.mul(p1,tf.log(p2)),tf.mul(1-p1,tf.log(1-p2))), [-1])
        AND = tf.add(AND1, AND2)
        eq = tf.reduce_sum(tf.abs(y - y_),reduction_indices=[1])
        ADD = tf.add(eq,rho*AND)
        cross_entropy = tf.reduce_mean(ADD)
    elif cost_fun == 'cusce':
        part1 = tf.slice(y_,[0,1],[-1,1])
        part2 = tf.slice(y,[0,1],[-1,1])
        AND1 = -tf.reshape(tf.add(tf.mul(part1,tf.log(part2)),tf.mul(1-part1,tf.log(1-part2))), [-1])
        p1 = tf.slice(y_,[0,4],[-1,1])
        p2 = tf.slice(y,[0,4],[-1,1])
        AND2 = -tf.reshape(tf.add(tf.mul(p1,tf.log(p2)),tf.mul(1-p1,tf.log(1-p2))), [-1])
        AND = tf.add(AND1, AND2)
        #eq = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
        eq = tf.nn.softmax_cross_entropy_with_logits(intermediate,y_)
        ADD = tf.add(eq,rho*AND)
        cross_entropy = tf.reduce_mean(ADD)
        #experiment = sess.run(cross_entropy, feed_dict={x: validation_features[:2], y_: validation_labels[:2]})
        #print "CE value: " + str(experiment)
    # function to minimize, the [1] indicates that the summation is done on the 2nd dimension
    #opt= tf.train.AdadeltaOptimizer(alpha)
    #opt = tf.train.AdagradOptimizer(alpha)
    opt = tf.train.AdamOptimizer(alpha)
    #opt = tf.train.GradientDescentOptimizer(alpha)
    #opt = tf.train.MomentumOptimizer(alpha, 0.25)
    #opt = tf.train.FtrlOptimizer(alpha)
    #opt = tf.train.RMSPropOptimizer(alpha)
    tvars = tf.trainable_variables()
    grads,_ = tf.clip_by_global_norm(tf.gradients(cross_entropy,tvars),clip_norm=clip_grad)
    optimizer = opt.apply_gradients(zip(grads, tvars))
    train_step = opt.minimize(cross_entropy)
    #train_step = tf.train.AdadeltaOptimizer(alpha).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(alpha).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    scores3_train = np.zeros(max_iter)
    scores3_val = np.zeros(max_iter)
    scores3_precision = np.zeros(max_iter)
    scores3_recall = np.zeros(max_iter)
    validation_scores = []
    target_score = 0
    acc_val = 0
    fp = 0
    fn = 0
    if pretrained == 0:
        ii = 0
        while ii <= stop:
            #Training for 1000 iterations
            if batch_size > 0:
                batch = random.randint(0,training_features.__len__() - batch_size - 1)
                forward_batch_xs, backward_batch_xs = forward_features[batch: batch + batch_size], backward_features[batch: batch + batch_size]
                batch_ys = training_labels[batch: batch + batch_size]
            else:
                forward_batch_xs, backward_batch_xs = forward_features, backward_features
                batch_ys = training_labels
            #assert training_features.__len__() == training_labels.__len__()
            reference = training_features[0][0].__len__()
            #print reference
            if debug_features == 1:
                for iii in range(1,training_features.__len__()):
                    if reference != training_features[iii].__len__():
                        print "element " + str(iii) + " is not equal"
                        #print str(training_features[iii]) + ", the reference is " + str(training_features[0])
                        print str(training_features[iii].__len__()) + " instead of " + str(reference)
                        #display_features(training_features[iii])
            #A batch of 100 random data points is chosen from the training set
            #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            run_model(sess, train_step, forward_x, forward_batch_xs, backward_x, backward_batch_xs, forward_pass, backward_pass, y_, batch_ys)

            # if (ii+1)%1000 == 0:
            #     with open(PATH + "weights_" + str(ii),'w') as f:
            #         ujson.dump(W,f)
            if ii%resolution == 0:
                #correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
                #Arg_max returns the index of the highest entry along some axis; tf.equal returns boolean data types
                #accuracy_validation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #Find the mean of the boolean predictions
                #acc_train = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                start_index = 0
                end_index = 7
                acc_train = run_model(sess, cross_entropy, forward_x, forward_batch_xs, backward_x, backward_batch_xs, forward_pass, backward_pass, y_, batch_ys)
                tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file,validation_features,sess,forward_x,backward_x,W,b,y, model)
                pairing = (matrix, acc_val)
                multiclass.append(pairing)
                summary_statement(ii, acc_train, acc_val, tp1, fp, tp2, fn, matrix)
                pickle.dump(multiclass, open("multiclass", "wb"))
                validation_scores.append(acc_val)
                target_score = np.max(validation_scores)
                if acc_val == 1:
                    print 'Done!!!'
                    while acc_val > 0:
                        infinite_loop = 1
            if ii == stop:
                tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file, validation_features, sess, forward_x,backward_x, W, b, y, model, final=1)
                if acc_val < target_score - 0.1:
                    stop += 10
                else:
                    save_model = 1
            if ii < scores3_train.__len__():
                scores3_train[ii] += acc_train
                scores3_val[ii] += acc_val
                scores3_precision[ii] += div(tp1, tp1 + fp)
                scores3_recall[ii] += div(tp2, tp2 + fn)
            ii += 1
    if predict == 1:
        if pretrained == 1:
            with open(PATH + pretrained_file) as f:
                feats = ujson.load(f)
        t1 = datetime.datetime.now()
        print "Starting predictions..."
        tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file,validation_features,sess,forward_x,backward_x,W,b,y, model, 1)
        if validation_scores.__len__() > 0:
            if acc_val < np.max(validation_scores) - 0.01:
                acc_val = np.max(validation_scores)
        summary_statement("Final", acc_train, acc_val, tp1, fp, tp2, fn, matrix)
        t2 = datetime.datetime.now()
        print "Predictions complete!!!"
        print "It took " + str(t2-t1) + " to complete the evaluation"
        write_predictions(predictions, test_file)
    return scores3_train, target_score, scores3_precision, scores3_recall




if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print "Starting program at " + str(t1)
    predict = 1
    if test == 1:
        num = 1
        splits = 1
    index1 = WORD2VEC.find('_')
    index2 = WORD2VEC.find('d',index1)
    DIM = int(WORD2VEC[index1+1:index2])
    print "Loading model...."
    if Google_vecs == 1:
        DIM = 300
        model = gensim.models.Word2Vec.load_word2vec_format(WORD2VEC_PATH + 'GoogleNews-vectors-negative300.bin.gz',
                                                            binary=True)
        print "GoogleNews-vectors loaded!!!"
        phrase = "Yangtze_River"
        if model.__contains__(phrase):
            print "The model contains " + str(phrase)
        phrase = "New_York"
        if model.__contains__(phrase):
            print "The model contains " + str(phrase)
        phrase = "new_york"
        if model.__contains__(phrase):
            print "The model contains " + str(phrase)
        phrase = "a_lot"
        if model.__contains__(phrase):
            print "The model contains " + str(phrase)
        phrase = "A_Lot"
        if model.__contains__(phrase):
            print "The model contains " + str(phrase)
    else:
        try:
            model = gensim.models.Word2Vec.load(WORD2VEC_PATH + WORD2VEC)
            print "Model loaded!!!"
        except:
            model = gensim.models.Word2Vec.load(WORD2VEC_PATH + WORD2VEC2)
            print "Model loaded!!!"
        #gensim.models.Word2Vec.
    scores1_train = np.zeros([num,max_iter])
    scores1_val = np.zeros([num,max_iter])
    index = range(1,max_iter+1)
    leg1 = []
    flag1 = 0
    flag1_error = 0
    for i in range(0, num):
        scores2_train = np.zeros([splits,max_iter])
        scores2_val = np.zeros([splits,max_iter])
        leg1.append('split_' + str(i) + '_train')
        leg1.append('split_' + str(i) + '_val')
        leg2 = []
        flag2 = 0
        flag2_error = 0
        for j in range(0,splits):
            if test == 1:
                print "Extracting features from dimsum16.train"
                training_features, training_labels = rnn_features.rnn_features(PATH + "dimsum16.train",WINDOW, FEAT_WIN, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
                remove_target, forward_pass, backward_pass, forward_reverse, backward_reverse,
                TOKENS_PATH + "dimsum16.train_vectors", LEMMAS_PATH + "dimsum16.train_vectors", uni_skip, bi_skip)
            elif sample == 1:
                print "Extracting features from fold_test"
                training_features, training_labels = rnn_features.rnn_features(PATH + "fold_test", WINDOW, FEAT_WIN, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
                remove_target, forward_pass, backward_pass, forward_reverse, backward_reverse,
                TOKENS_PATH + "fold_test_vectors", LEMMAS_PATH + "fold_test_vectors", uni_skip, bi_skip)
            else:
                print "Extracting features from fold_" + str(i+1) + str(j+1) + "_train"
                training_features, training_labels = rnn_features.rnn_features(PATH + "fold_" + str(i+1) + str(j+1) + "_train",
                WINDOW, FEAT_WIN, POS_WINDOW, TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, tag_features,
                token_features, lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas,
                normalize, debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg,
                sentence_embedding, gen_vec, unk, unknown_token, remove_target, forward_pass, backward_pass, forward_reverse, backward_reverse,
                TOKENS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors",
                LEMMAS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors", uni_skip, bi_skip)
            print "Extracting validation data..."
            if test == 1:
                print "Extracting features from dimsum16.test.blind"
                validation_features, validation_labels = rnn_features.rnn_features(PATH + "dimsum16.test.blind", WINDOW, FEAT_WIN, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,1, ngram, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
                remove_target, forward_pass, backward_pass, forward_reverse, backward_reverse,
                TOKENS_PATH + "dimsum16.test.blind_vectors", LEMMAS_PATH + "dimsum16.test.blind_vectors", uni_skip, bi_skip)
            elif sample == 1:
                print "Extracting features from fold_test"
                validation_features, validation_labels  = rnn_features.rnn_features(PATH + "fold_test", WINDOW, FEAT_WIN, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,1, ngram, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
                remove_target, forward_pass, backward_pass, forward_reverse, backward_reverse,
                TOKENS_PATH + "fold_test_vectors", LEMMAS_PATH + "fold_test_vectors", uni_skip, bi_skip)
            else:
                print "Extracting features from fold_" + str(i + 1) + str(j + 1) + "_test"
                validation_features, validation_labels = rnn_features.rnn_features(PATH + "fold_" + str(i+1) + str(j+1) + "_test",
                WINDOW, FEAT_WIN, POS_WINDOW, TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,1, ngram, tag_features,
                token_features, lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas,
                normalize, debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg,
                sentence_embedding, gen_vec, unk, unknown_token, remove_target, forward_pass, backward_pass, forward_reverse, backward_reverse,
                TOKENS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors", LEMMAS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors",
                uni_skip, bi_skip)
            print "No of training instances: " + str(training_features.__len__())
            print "No of validation instances: " + str(validation_features.__len__())
            scores3_train = np.zeros([repetitions,max_iter])
            scores3_val = np.zeros([repetitions,max_iter])
            leg2.append('split_' + str(j) + '_train')
            leg2.append('split_' + str(j) + '_val')
            leg3 = []
            flag3 = 0
            for rep in range(0,repetitions):
                print "Training Neural Network..."
                leg3.append('rep_' + str(rep) + '_train')
                leg3.append('rep_' + str(rep) + '_val')
                if test == 1:
                    train_score, val_score, precision, recall = train_RNN(training_features,training_labels,validation_features,
                    validation_labels,tag_set.__len__(),cost_function, learning_rate,max_iter,predict, rho,
                    batch_size, PATH + "dimsum16.test.blind", model, hidden)
                elif sample == 1:
                    train_score, val_score, precision, recall = train_RNN(training_features, training_labels, validation_features,
                    validation_labels, tag_set.__len__(), cost_function, learning_rate, max_iter, predict, rho,
                    batch_size, PATH + "fold_test", model, hidden)
                else:
                    train_score, val_score, precision, recall = train_RNN(training_features,training_labels,validation_features,
                    validation_labels,tag_set.__len__(),cost_function,learning_rate,max_iter,predict, rho,
                    batch_size, PATH + "fold_" + str(i+1) + str(j+1) + "_test", model, hidden)
                scores3_train[rep] += train_score
                scores3_val[rep] += val_score
                #flag3 = plot_scores(index, train_score, val_score, precision, recall, 1, leg3, flag3)
            scores2_train[j] += np.mean(scores3_train,axis=0)
            scores2_val[j] += np.mean(scores3_val,axis=0)
            #flag2 = plot_scores(index,scores2_train[j],scores2_val[j],100*i + 2,leg2,flag2)
            #flag2_error = plot_error_bars(index,scores2_train[j],np.std(scores3_train,axis=0),scores2_val[j],np.std(scores3_val,axis=0),100*i + 12,leg2,flag2_error)
        scores1_train[i] += np.mean(scores2_train,axis=0)
        scores1_val[i] += np.mean(scores2_val,axis=0)
        #flag1 = plot_scores(index,scores1_train[i],scores1_val[i],1,leg1,flag1)
        #flag1_error = plot_error_bars(index,scores1_train[i],np.std(scores2_train,axis=0),scores1_val[i],np.std(scores2_val,axis=0),11,leg1,flag1_error)
    final_score_train_mean = np.mean(scores1_train,axis=0)
    final_score_train_std = np.std(scores1_train,axis=0)
    final_score_val_mean = np.mean(scores1_val,axis=0)
    final_score_val_std = np.std(scores1_val,axis=0)
    #flag = plot_scores(index,final_score_train_mean,final_score_val_mean,1,leg1,0)
    #flag_error = plot_error_bars(index,final_score_train_mean,final_score_train_std,final_score_val_mean,final_score_val_std,11,leg1,0)
    t2 = datetime.datetime.now()
    print "The program ended at " + str(t2)
    print "It took " + str(t2 - t1) + " to finish the evaluation"
