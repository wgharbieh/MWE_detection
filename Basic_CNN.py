DIR = "/home/waseem/"

WORD2VEC_PATH = DIR + "models/"
PATH = DIR + "Downloads/dimsum-data-1.5/"
model_path = PATH + "models/"
char_seq2seq_path = model_path + "char_seq2seq/"
model_name = "NN_model"

import tensorflow as tf
import gensim
import numpy as np
import datetime
import ujson
import extract_mwes
import sys
import rnn_features as features
import operator
import error_analysis
import pickle
import string
from lstm import LSTMCell, BNLSTMCell, orthogonal_initializer, batch_norm
import optimize_parameters
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import multiprocessing
from char_seq2seq import char_seq2seq, char_seq
import fasttext

lower_bound = 1
upper_bound = 100 - lower_bound
#scaler = RobustScaler(quantile_range= (lower_bound, upper_bound))
#scaler = MinMaxScaler()
scaler = StandardScaler()
imputer = Imputer(missing_values=np.nan, strategy="mean")
feat_preprocess = []
imputing = []

scale_data = 0
impute_data = 0
fasttext_vectors = 0

FOLD = "11"
WINDOW = 1
POS_WINDOW = 1
TAG_WINDOW = 1
MWE_WINDOW = 1
FEAT_WIN = 1
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
hidden = [(800,'tanh'),(6,'soft')]
conv_nn = [([1,5,5,32],[1,1,1,1]),([1,2,2,1],[1,2,2,1])]
#hidden = None
test = 1
max_iter = 10000
reg = 1e10
sentence_embedding = 0
learning_rate = 1e-2
gen_vec = 0
unk = 0
remove_target = 0
normalize = 0
sample = 0
### Batch Parameters
batch_splits = 0
if sample == 1:
    model_name += "_sample"
    batch_splits = 0
normalize_batch = 0
epsilon = 1e-3
###################################
uni_skip = 0
bi_skip = 0
### Word2vec Parameters
Google_vecs = 0
tweet = 0
##############################
con_win = 100
poshmm = 0
top_k_predictions = 3
char_win = 3
gram_win = 3
seed_val = 101
drop_out = 0.5


decay_rate = 1
#decay_rate = 1 - 1e-3
layer1_orthogonal = 0
layer2_orthogonal = 0
layer3_orthogonal = 0
BN_decay = 0.9999

purely_random = 1
drop_inputs = 0
realistic = 1

decay_steps = 100
learning_decay_rate = 1
staircase = True

##################################
genetic_algorithm = 0
if genetic_algorithm == 1:
    model_path += "_GA"
initial_population = 8
total_population = 100
threshold_error = 0.05
generations = 3
mutation_rate = 1e-3
probabilistic = 0
training_set_factor = 0.5
testing_set_factor = 0.5

###################################

word_shape = 0
normalize_cap = 0
buckets = [(5,5),(10,10),(15,15),(20,20),(25,25),(30,30),(35,35)]
char_seq2seq_hidden = 25 #int(sys.argv[1])
char_seq2seq_layers = 1
max_gradient_norm = 1
batch_size = 0
learning_rate_decay_factor = 1
use_lstm = True
ADAM = True
stop_seq2seq = 10 #34902
seq2seq_decay_steps = 1000
seq2seq_staircase = False
cus_LSTM = 1
char_drop = 1
backward_seq2seq = 0
orthogonal_seq2seq = 1
with_start = 0
seq2seq = 0
cross_val_eval = 1
evaluate = 1
char_file = "char_seq_25_saved"
pretrained_chars = 1

#############################################################



def parse_input(x, var_type):
    #print "x: " + str(x)
    if x.__contains__(','):
        k = x.split(',')
        y = ''.join(k)
        #print "y: " + str(y)
        t = ()
        for element in y:
            if var_type == 0:
                t += (int(element),)
            else:
                t += (float(element),)
        return t
    if var_type == 0:
        return int(x)
    else:
        return float(x)

# FOLD = sys.argv[1]
# WINDOW = parse_input(sys.argv[2],0)
# POS_WINDOW = parse_input(sys.argv[3],0)
# TAG_WINDOW = parse_input(sys.argv[4],0)
# MWE_WINDOW = parse_input(sys.argv[5],0)
# DIM = int(sys.argv[6])
# embedding_window = int(sys.argv[7]) #4
# tag_features = int(sys.argv[8])
# lemma_features = int(sys.argv[9])
# pos_features = int(sys.argv[10])
# context = int(sys.argv[11])
# in_mwe = int(sys.argv[12])
# ngram = int(sys.argv[13])
# rho = int(sys.argv[14])
# cost_function = sys.argv[15] #sce
# hidden = [(int(sys.argv[16]),'tanh'),(6,'soft')]
# test = int(sys.argv[17])
# max_iter = int(sys.argv[18])
# sentence_embedding = 0
# learning_rate = 1e-2
# remove_target = int(sys.argv[19])
# normalize = int(sys.argv[20])
# seed_val = int(sys.argv[21])
# drop_out = parse_input(sys.argv[22],1)
# sample = int(sys.argv[23])
# generations = int(sys.argv[24])
# mutation_rate = float(sys.argv[25])
# probabilistic = int(sys.argv[26])
# train_set_factor = float(sys.argv[27])
# test_set_factor = float(sys.argv[28])

avg_tokens = 1
avg_lemmas = 1
cbow = 0
mincount = 15
if fasttext_vectors == 0:
    if tweet == 0:
        WORD2VEC = "Word2Vec_" + str(DIM) + "d_" + str(embedding_window) + "window_" + str(mincount) + "mincount_5nsampling_"
        if cbow == 1:
            WORD2VEC += "cbow"
        else:
            WORD2VEC += "skipgram"
        if unk == 1:
            WORD2VEC += '_UNK'
        WORD2VEC += ".bin"
        WORD2VEC2 = "wikiEmbeddingsStanfordTokenizedSkipGram" + str(embedding_window) + "-dimension" + str(DIM) + "vectors.bin"
    else:
        WORD2VEC = "TweetEmbeddingsSkipGram-window5-dimension100vectors.bin"
else:
    FASTTEXT = "fasttext"
    skip = 1
    if skip == 0:
        FASTTEXT += "_cbow"
    else:
        FASTTEXT += "_skip"
    embedding_window = 2
    FASTTEXT += "_dim" + str(DIM) + "_cw" + str(embedding_window)
    epoch = 5
    FASTTEXT += "_epoch" + str(epoch)
    mincount = 5
    FASTTEXT += "_mincount" + str(mincount)
    ns = 5
    FASTTEXT += "_ns" + str(ns)
    word_ngrams = 1
    FASTTEXT += "_maxw" + str(word_ngrams)
    minchar_ngrams = 3
    FASTTEXT += "_minc" + str(minchar_ngrams)
    maxchar_ngrams = 6
    FASTTEXT += "_maxc" + str(maxchar_ngrams) + ".bin"
TOKENS_PATH = PATH + 'tokens/'
LEMMAS_PATH = PATH + 'lemmas/'
unknown_token = 'UNK'
splits = 5
num = 1
repetitions = 1
 #400
resolution = 100
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
np.random.seed(seed_val)


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01, seed=seed_val)
    #return tf.Variable(tf.concat(0, [initial1, initial2]))
    return tf.Variable(initial, name=name)
    #Initialize with small weights


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape= shape)
    return tf.Variable(initial, name=name)
    # Initialize with small positive weights

def conv2d(x, W, stride, pad = 'SAME'):
  return tf.nn.conv2d(x, W, strides=stride, padding=pad)

def max_pool_2x2(x, window, stride, pad = 'SAME'):
  return tf.nn.max_pool(x, ksize=window,
                        strides=stride, padding=pad)

def tag_feat(tag,tag_set):
    feats = []
    for ts in tag_set:
        if ts == tag:
            feats.append(1)
        else:
            feats.append(0)
    return feats


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


def get_score(predictions,labels):
    if predictions.__len__() != labels.__len__():
        print "Predictions and labels are of not equal length"
    correct = 0
    for i in range(0,labels.__len__()):
        index = tag_set.index(predictions[i])
        if labels[i][index] == 1:
            correct += 1
    return correct*1.0/labels.__len__()


def get_inside(tag):
    inside = 0
    if tag == 0:
        inside = 1 # 'B', 'i', and 'O' are not allowed
    elif tag == 1:
        inside = 2 # 'B' is not allowed
    elif tag == 2:
        inside = 0 # Only 'B' and 'O' are allowed
    elif tag == 3:
        inside = 3 # Only 'i' is allowed
    elif tag == 4:
        inside = 4 # 'i','o', and 'I' are allowed
    elif tag == 5:
        inside = 5 # 'I' and 'b' are allowed
    else:
        print "Inside does not exist!!"
    return inside


def get_possible_tags(inside):
    # [B,I,O,b,i,o]
    # [0,1,2,3,4,5]
    if inside == 0:
        return [0,2]
    if inside == 1:
        return [1,3,5]
    if inside == 2:
        return [1,2,3,4,5]
    if inside == 3:
        return [4]
    if inside == 4:
        return [1,4,5]
    if inside == 5:
        return [1,3]


def amend_pred(pred,inside):
    cl = -1
    possible_tags = get_possible_tags(inside)
    max_tag = -1
    for i in possible_tags:
        if pred[i] > max_tag:
            temp = np.zeros(tag_set.__len__()).tolist()
            temp[i] = 1
            cl = temp
            max_tag = pred[i]
            inside = get_inside(i)
    return cl, inside


def Viterbi(sequence, tag_set, emission_prob, ngram):
    new_sequence = []
    tag_dict= []
    for i in range(0,sequence.__len__()):
        tag_dict.append({})
    bp = {}
    prev_tag = ()
    for i in range(0,ngram):
        prev_tag += (start_tag,)
    temp_dict = {}
    if debug_viterbi == 1:
        print "emission_prob: " + str(emission_prob)
    for tag in tag_set:
        current_tuple = prev_tag[1:] + (tag,)
        if debug_viterbi == 1:
            print "current_tuple: " + str(current_tuple)
        if emission_prob.__contains__(current_tuple):
            temp_dict[current_tuple] = emission_prob[current_tuple]*sequence[0][tag_set.index(tag)]
            bp[(0,current_tuple)] = prev_tag
    if debug_viterbi == 1:
        print "temp_dict: " + str(temp_dict)
    tag_dict[0] = temp_dict
    if debug_viterbi == 1:
        print "sequence[0] = " + str(sequence[0])
        print "tag_dict: " + str(tag_dict)
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


def get_tags(sequence, tag_set):
    #print sequence
    tags = []
    for element in sequence:
        #print "element: " + str(element)
        index = np.argmax(element)
        tags.append(tag_set[index])
    return tags


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


def div(x,y):
    val = 0
    if x != 0 or y != 0:
        val = x*1.0/y
    return val

def get_fscore(precision,recall):
    fscore = 0
    if precision != 0 and recall != 0:
        fscore = 2*precision*recall/(precision + recall)
    return fscore


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


def get_max_predictions(new_sequences):
    max_arr = []
    for max_seq in new_sequences:
        for i in range(1, max_seq.__len__() - 1):
            max_arr.append(max_seq[i])
    return max_arr

def get_pred(test_file,validation_features,sess,x,W,b,y, keep_prob, training, emission_arr, pos_tags, model = None,final = 0):
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
            modified_token = features.preprocess(token)
            if model.__contains__(modified_token):
                model_token = modified_token
            elif model.__contains__(modified_token.lower()):
                model_token = modified_token.lower()
            else:
                model_token = unknown_token
            token_sent.append(model_token)
            lemma = k[2]
            modified_lemma = features.preprocess(lemma)
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
            if tag_features == 1:
                for p in prev_tags:
                    tag_vector = tag_feat(p, tag_set)
                    validation_features[j][0:6] = tag_vector
##run_model(sess, parameter, x, batch_xs, drop_out, keep_prob, batch_normalization, training, is_training = None, y_ = None, batch_ys = None)
                    pred = run_model(sess, y, x, [validation_features[j]], drop_out, keep_prob, normalize_batch, training, False)
                    update_sequences(tag_sequences, p, pred[0], emission_arr[0], new_sequences, tag_set)
                tag_sequences = new_sequences
                prev_tags = tag_set
            else:
                pred = run_model(sess, y, x, [validation_features[j]], drop_out, keep_prob, normalize_batch, training, False)
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
                    #print "emission_arr: " + str(emission_arr)
                    max_predictions = Viterbi(sequence, tag_set, emission_arr[0], ngram)
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


def create_random_indices(input_size, drop_inputs):
    indices = []
    remaining = int(np.floor(drop_inputs*input_size))
    for i in range(input_size - 1, -1, -1):
        if remaining == 0:
            break
        if i > remaining:
            toss = np.random.uniform(0,1)
            if toss < drop_inputs:
                indices.append(i)
                remaining -= 1
        else:
            indices.append(i)
            remaining -= 1
    #print "random_indices: " + str(indices)
    return indices


def zero_out(inputs, drop_inputs, window, tag_features, word_dim, extra_features_size):
    indices = create_random_indices(inputs.__len__(), drop_inputs)
    #print "indices: " + str(indices)
    new_batch = []
    #print "inputs.__len__(): " + str(inputs.__len__())
    for i in range(0, inputs.__len__()):
        new_batch.append(inputs[i])
    #print "new_batch.__len__(): " + str(new_batch.__len__())
    for index in indices:
        feat_chunk = (window*2 + 1)
        start_index = 6 * tag_features
        tokens_lemmas = feat_chunk*2
        #print "inputs[" + str(index) + "]" + " = " + str(new_batch[index][start_index: start_index + (tokens_lemmas*(word_dim + extra_features_size))])
        drops = np.random.randint(0, tokens_lemmas)
        #print "drops: " + str(drops)
        indices2 = create_random_indices(tokens_lemmas, drops*1.0/tokens_lemmas)
        if realistic == 1:
            for i in indices2:
                if i >= feat_chunk:
                    new_index = i - feat_chunk
                    if not indices2.__contains__(new_index):
                        indices2.append(new_index)

        for i in indices2:
            si = start_index + (word_dim + extra_features_size)*i
            ei = si + word_dim
            #print "From: " + str(si) + " to " + str(ei)
            new_batch[index][si:ei] = np.zeros(word_dim)
        #print "indices2: " + str(indices2)
        #print "index: " + str(index)
        #print "inputs[" + str(index) + "]" + " = " + str(new_batch[index][start_index: start_index + (tokens_lemmas * (word_dim + extra_features_size))])
    return new_batch

def run_model(sess, parameter, x, batch_xs, drop_out, keep_prob, batch_normalization, training, is_training = None, y_ = None, batch_ys = None):
    if batch_normalization == 1:
        if drop_out < 1:
            if is_training == False:
                return sess.run(parameter, feed_dict={x: batch_xs, keep_prob: 1.0, training: is_training})
            else:
                new_batch = zero_out(batch_xs, drop_inputs, WINDOW, tag_features, DIM, 12)
                return sess.run(parameter, feed_dict={x: new_batch, keep_prob: drop_out, training: is_training})
        else:
            if is_training == False:
                return sess.run(parameter, feed_dict={x: batch_xs, training: is_training})
            else:
                new_batch = zero_out(batch_xs, drop_inputs, WINDOW, tag_features, DIM, 12)
                return sess.run(parameter, feed_dict={x: new_batch, y_: batch_ys, training: is_training})
    elif y_ == None:
        if drop_out < 1:
            return sess.run(parameter, feed_dict={x: batch_xs, keep_prob: 1.0})
        else:
            return sess.run(parameter, feed_dict={x: batch_xs})
    else:
        if drop_out < 1:
            #new_batch = zero_out(batch_xs, drop_inputs, WINDOW, tag_features, DIM, 12)
            return sess.run(parameter, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: drop_out})
        else:
            #new_batch = zero_out(batch_xs, drop_inputs, WINDOW, tag_features, DIM, 12)
            return sess.run(parameter, feed_dict={x: batch_xs, y_: batch_ys})

def train_NN(training_file, training_features,training_labels, validation_features, validation_labels, emission_arr, pos_tags, classes, cost_fun, alpha, stop,
             predict, rho, batch_size, test_file, model = None, hidden = None):
    acc_train = 0
    acc_val = 0
    input_size = training_features[0].__len__()
    with tf.name_scope('inputs'):
        x_pre = tf.placeholder(tf.float32, [None, input_size], name='x_input')
        x = tf.reshape(x_pre, )
        y_ = tf.placeholder(tf.float32, [None, classes], name='y_input')
    #x_input = [x]
    #Holds variables, 'None' indicates unknown length
    conv_weights1, conv_stride1 = conv_nn[0][0], conv_nn[0][1]
    pool_window1, pool_stride1 = conv_nn[1][0], conv_nn[1][1]
    W_conv1 = weight_variable(conv_weights1)
    b_conv1 = bias_variable([conv_weights1[3]])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, conv_stride1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1, pool_window1, pool_stride1)

    h_pool = h_pool1
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    layer1 = 'layer1'
    layer2 = 'layer2'
    layer3 = 'layer3'
    if hidden == None:
        l1 = classes
        with tf.name_scope('weights_1'):
            if layer1_orthogonal == 1:
                W = tf.get_variable('W', [input_size, l1], initializer=orthogonal_initializer(), name='orth_W1')
            else:
                W = weight_variable([input_size,l1], name='W1')
            tf.histogram_summary(layer1 + "/weights", W)

        #This is how a variable is declared, the model parameters should be declared as variables
            b = bias_variable([l1], name='b1')
            tf.histogram_summary(layer1 + "/biases", b)
        #y1 = tf.nn.relu(tf.matmul(x,W) + b)
        intermediate = tf.matmul(h_pool,W) + b
        with tf.name_scope(layer1):
            y1 = tf.nn.softmax(intermediate, name='activation1')
            tf.histogram_summary(layer1 + "/activations", y1)
        y = y1
    elif hidden.__len__() == 2:
        l1 = hidden[0][0]
        with tf.name_scope('weights_1'):
            if layer1_orthogonal == 1:
                W1 = tf.get_variable('W1', [input_size, l1], initializer=orthogonal_initializer(), name='orth_W1')
            else:
                W1 = weight_variable([input_size,l1], name='W1')
        #This is how a variable is declared, the model parameters should be declared as variables

            tf.histogram_summary(layer1 + "/weights", W1)
        # w2_BN = tf.Variable(w2_initial)
        # z2_BN = tf.matmul(l1_BN, w2_BN)
        # batch_mean2, batch_var2 = tf.nn.moments(z2_BN, [0])
        # scale2 = tf.Variable(tf.ones([100]))
        # beta2 = tf.Variable(tf.zeros([100]))
        # BN2 = tf.nn.batch_normalization(z2_BN, batch_mean2, batch_var2, beta2, scale2, epsilon)
        # l2_BN = tf.nn.sigmoid(BN2)

            if normalize_batch == 1:
                z1_BN = tf.matmul(h_pool, W1)
                #batch_mean1, batch_var1 = tf.nn.moments(z1_BN, [0])
                #scale1 = tf.Variable(tf.ones([l1]))
                #beta1 = tf.Variable(tf.zeros([l1]))
                BN1 = batch_norm(z1_BN, 'layer1_BN', training, decay=BN_decay)
                z1 = tf.nn.sigmoid(BN1)
            else:
                b1 = bias_variable([l1], name='b1')
                z1 = tf.matmul(h_pool,W1) + b1
            tf.histogram_summary(layer1 + "/biases", b1)
        activation = hidden[0][1]
        with tf.name_scope('layer_1'):
            if activation == 'tanh':
                y1 = tf.nn.tanh(z1, name='activation1')
            elif activation == 'sig':
                y1 = tf.nn.sigmoid(z1, name='activation1')
            elif activation == 'relu':
                y1 = tf.nn.relu(z1, name='activation1')
            elif activation == 'soft':
                y1 = tf.nn.softmax(z1, name='activation1')
            y1_final = y1
            if drop_out < 1:
                h_fc1_drop = tf.nn.dropout(y1, keep_prob, name='dropout1')
                y1_final = h_fc1_drop
            tf.histogram_summary(layer1 + "/activations", y1_final)
        l2 = hidden[1][0]
        with tf.name_scope('weights_2'):
            if layer2_orthogonal == 1:
                W2 = tf.get_variable('W2', [l1,l2], initializer=orthogonal_initializer(), name='orth_W2')
            else:
                W2 = weight_variable([l1,l2], name='W2')
        #This is how a variable is declared, the model parameters should be declared as variables
            tf.histogram_summary(layer2 + "/weights", W2)
            b2 = bias_variable([l2], name='b2')
            tf.histogram_summary(layer2 + "/biases", b2)
        activation = hidden[1][1]
        intermediate = tf.matmul(y1_final,W2) + b2
        with tf.name_scope('layer_2'):
            if activation == 'tanh':
                y2 = tf.nn.tanh(intermediate, name='activation2')
            elif activation == 'sig':
                y2 = tf.nn.sigmoid(intermediate, name='activation2')
            elif activation == 'relu':
                y2 = tf.nn.relu(intermediate, name='activation2')
            elif activation == 'soft':
                y2 = tf.nn.softmax(intermediate, name='activation2')
            tf.histogram_summary(layer2 + "/activations", y2)
        y = y2
    elif hidden.__len__() == 3:
        l1 = hidden[0][0]
        with tf.name_scope('weights_1'):
            if layer1_orthogonal == 1:
                W1 = tf.get_variable('W1', [input_size,l1], initializer=orthogonal_initializer(), name='orth_W1')
            else:
                W1 = weight_variable([input_size,l1], name='W1')
            #This is how a variable is declared, the model parameters should be declared as variables
            tf.histogram_summary(layer1 + "/weights", W1)
            if normalize_batch == 1:
                z1_BN = tf.matmul(h_pool, W1)
                BN1 = batch_norm(z1_BN, 'layer1_BN', training, decay=BN_decay)
                z1 = tf.nn.sigmoid(BN1)
            else:
                b1 = bias_variable([l1], name='b1')
                z1 = tf.matmul(h_pool,W1) + b1
            tf.histogram_summary(layer1 + "/biases", b1)
        activation = hidden[0][1]
        with tf.name_scope('layer_1'):
            if activation == 'tanh':
                y1 = tf.nn.tanh(z1, name='activation1')
            elif activation == 'sig':
                y1 = tf.nn.sigmoid(z1, name='activation1')
            elif activation == 'relu':
                y1 = tf.nn.relu(z1, name='activation1')
            elif activation == 'soft':
                y1 = tf.nn.softmax(z1, name='activation1')
            y1_final = y1
            if drop_out < 1:
                h_fc1_drop = tf.nn.dropout(y1, keep_prob, name='dropout1')
                y1_final = h_fc1_drop
            tf.histogram_summary(layer1 + "/activations", y1_final)
        l2 = hidden[1][0]
        with tf.name_scope('weights_2'):
            if layer2_orthogonal == 1:
                W2 = tf.get_variable('W2', [l1,l2], initializer=orthogonal_initializer(), name='orth_W2')
            else:
                W2 = weight_variable([l1,l2], name='W2')
            #This is how a variable is declared, the model parameters should be declared as variables
            tf.histogram_summary(layer2 + "/weights", W2)
            if normalize_batch == 1:
                z2_BN = tf.matmul(y1_final, W2)
                BN2 = batch_norm(z2_BN, 'layer2_BN', training, decay=BN_decay)
                z2 = tf.nn.sigmoid(BN2)
            else:
                b2 = bias_variable([l2], name='b2')
            tf.histogram_summary(layer2 + "/biases", b2)
            z2 = tf.matmul(h_pool,W2) + b2
        activation = hidden[1][1]
        intermediate1 = z2
        with tf.name_scope('layer_2'):
            if activation == 'tanh':
                y2 = tf.nn.tanh(intermediate1, name='activation2')
            elif activation == 'sig':
                y2 = tf.nn.sigmoid(intermediate1, name='activation2')
            elif activation == 'relu':
                y2 = tf.nn.relu(intermediate1, name='activation2')
            elif activation == 'soft':
                y2 = tf.nn.softmax(intermediate1, name='activation2')
            y2_final = y2
            if drop_out < 1:
                h_fc2_drop = tf.nn.dropout(y2, keep_prob, name='dropout2')
                y2_final = h_fc2_drop
            tf.histogram_summary(layer2 + "/activations", y2_final)
        l3 = hidden[2][0]
        with tf.name_scope('weights_3'):
            if layer3_orthogonal == 1:
                W3 = tf.get_variable('W3', [l2,l3], initializer=orthogonal_initializer(), name='orth_W3')
            else:
                W3 = weight_variable([l2,l3], name='W3')
            tf.histogram_summary(layer3 + "/weights", W3)
            #This is how a variable is declared, the model parameters should be declared as variables
            b3 = bias_variable([l3], name='b3')
            tf.histogram_summary(layer3 + "/biases", b3)
        activation = hidden[2][1]
        intermediate = tf.matmul(y2_final,W3) + b3
        with tf.name_scope('layer_3'):
            if activation == 'tanh':
                y3 = tf.nn.tanh(intermediate, name='activation3')
            elif activation == 'sig':
                y3 = tf.nn.sigmoid(intermediate, name='activation3')
            elif activation == 'relu':
                y3 = tf.nn.relu(intermediate, name='activation3')
            elif activation == 'soft':
                y3 = tf.nn.softmax(intermediate, name='activation3')
            tf.histogram_summary(layer3 + "/activations", y3)
        y = y3
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    #print validation_labels[:100]
    with tf.name_scope('loss'):
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
            start = 0
            end = 7
            #print [validation_labels[start:end]]
            #print str(sess.run(y,feed_dict={x:validation_features[start:end]}))
            y_B = tf.slice(y_,[0,0],[-1,1])
            #print "part1a: " + str(sess.run(part1a, feed_dict={y_:validation_labels[start:end]}))
            #part1b = tf.slice(y_, [0, 3], [-1, 3])
            #print "part1b: " + str(sess.run(part1b, feed_dict={y_: validation_labels[start:end]}))
            yB = tf.slice(y, [0, 0], [-1, 1])
            yO = tf.slice(y,[0,2],[-1,1])
            #print "part2: " + str(sess.run(part2, feed_dict={x: validation_features[start:end]}))

            factor = tf.greater(yO, yB)
            #print "factor: " + str(sess.run(factor, feed_dict={x: validation_features[start:end], y_: validation_labels[start:end]}))
            #AND1 = tf.reduce_sum(tf.mul(y_B, yO), reduction_indices=1)
            AND1 = tf.reduce_sum(tf.mul(y_B, yO)*tf.cast(factor,dtype=tf.float32), reduction_indices=1)
            #AND1 = tf.transpose(y_B * tf.cast(factor,dtype=tf.float32))
            #print "y: " + str(sess.run(y, feed_dict={x: validation_features[start:end], y_: validation_labels[start:end]}))
            #print "y_: " + str(validation_labels[start:end])
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
            print str(sess.run(y,feed_dict={x:validation_features[:2]}))
            part1 = tf.slice(y_,[0,0],[-1,1])
            part2 = tf.slice(y,[0,2],[-1,1])
            AND = tf.reshape(tf.mul(part1,part2), [-1])
            #eq1 = -tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1])
            #eq2 = -tf.reduce_sum((1-y_) * tf.log(1-y), reduction_indices=[1])
            eq1 = tf.nn.softmax_cross_entropy_with_logits(intermediate,y_)
            eq2 = tf.nn.softmax_cross_entropy_with_logits(1-intermediate, 1-y_)
            eq = tf.add(eq1, eq2)
            eq_num = sess.run(eq, feed_dict={x: validation_features[0:2], y_: validation_labels[:2]})
            print "y and y_ are equal? " + str(eq_num)
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
        validation_fscore = tf.placeholder(tf.float32)
        training_summary = tf.scalar_summary("training_loss", cross_entropy)
        validation_summary = tf.scalar_summary("validation_loss", cross_entropy)
    # function to minimize, the [1] indicates that the summation is done on the 2nd dimension
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    global_step = tf.Variable(0, trainable=False)
    if learning_decay_rate < 1:
        alpha_decay = tf.train.exponential_decay(alpha, global_step, decay_steps, learning_decay_rate, staircase)
    else:
        alpha_decay = alpha
    with tf.name_scope('train'):
        if learning_decay_rate < 1:
            train_step = tf.train.AdamOptimizer(alpha_decay).minimize(cross_entropy, global_step=global_step)
        else:
            train_step = tf.train.AdamOptimizer(alpha_decay).minimize(cross_entropy)
        tf.scalar_summary('alpha', alpha_decay)
        #train_step = tf.train.AdadeltaOptimizer(alpha).minimize(cross_entropy)
        #train_step = tf.train.AdagradOptimizer(alpha).minimize(cross_entropy)
        #train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)
        if decay_rate < 1:
            ema = tf.train.ExponentialMovingAverage(decay= decay_rate)
            if hidden.__len__() == 1:
                vars_op = [W1, b1]
            elif hidden.__len__() == 2:
                vars_op = [W1, b1, W2, b2]
            elif hidden.__len__() == 3:
                vars_op = [W1, b1, W2, b2, W3, b3]
            maintain_averages_op = ema.apply(vars_op)
            with tf.control_dependencies([train_step]):
                training_op = tf.group(maintain_averages_op)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.Session()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/", sess.graph)
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
    batch_size = 0
    population_weights = []
    population_biases = []
    fscores = []
    counter = initial_population
    if batch_splits > 1:
        batch_size = int(np.floor(training_features.__len__() * 1.0 / batch_splits))
    if pretrained == 0:
        ii = 0
        while ii <= stop:
            #Training for 1000 iterations
            if batch_size > 0:
                if purely_random == 1:
                    batch_xs = []
                    batch_ys = []
                    indices = create_random_indices(training_features.__len__(), 1.0/batch_splits)
                    for some_index in indices:
                        batch_xs.append(training_features[some_index])
                        batch_ys.append(training_labels[some_index])
                else:
                    batch = np.random.randint(0, training_features.__len__() - batch_size)
                    batch_xs, batch_ys = training_features[batch: batch + batch_size], training_labels[batch: batch + batch_size]
            else:
                batch_xs, batch_ys = training_features, training_labels
            assert training_features.__len__() == training_labels.__len__()
            reference = training_features[0].__len__()
            #print reference
            if debug_features == 1:
                for iii in range(1,training_features.__len__()):
                    if reference != training_features[iii].__len__():
                        print "element " + str(iii) + " is not equal"
                        #print str(training_features[iii]) + ", the reference is " + str(training_features[0])
                        print str(training_features[iii].__len__()) + " instead of " + str(reference)
                        #display_features(training_features[iii])
            #A batch of 100 random data points is chosen from the training set
            #run_model(sess, parameter, x, batch_xs, drop_out, keep_prob, y_ = None, batch_ys = None)
            if decay_rate < 1:
                #run_model(sess, parameter, x, batch_xs, drop_out, keep_prob, batch_normalization, training, is_training = None, y_ = None, batch_ys = None)
                train_acc, train_summ = run_model(sess, [training_op, training_summary], x, batch_xs, drop_out,
                                                  keep_prob, normalize_batch, training, True, y_, batch_ys)
            else:
                train_acc, train_summ = run_model(sess, [train_step, training_summary], x, batch_xs, drop_out,
                                                  keep_prob, normalize_batch, training, True, y_, batch_ys)
            writer.add_summary(train_summ, ii + 1)
            val_summ = run_model(sess, validation_summary, x, validation_features, drop_out,
                                 keep_prob, normalize_batch, training, True, y_, validation_labels)
            writer.add_summary(val_summ, ii + 1)
            summ = run_model(sess, merged, x, batch_xs, drop_out, keep_prob, normalize_batch, training, True, y_, batch_ys)
            writer.add_summary(summ, ii + 1)
            # if (ii+1)%1000 == 0:
            #     with open(PATH + "weights_" + str(ii),'w') as f:
            #         ujson.dump(W,f)
            if ii%resolution == 0:
                #correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
                #Arg_max returns the index of the highest entry along some axis; tf.equal returns boolean data types
                #accuracy_validation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #Find the mean of the boolean predictions
                #acc_train = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
##run_model(sess, parameter, x, batch_xs, drop_out, keep_prob, batch_normalization, training, is_training = None, y_ = None, batch_ys = None)
                acc_train = run_model(sess, cross_entropy, x, batch_xs, drop_out, keep_prob, normalize_batch, training, True, y_, batch_ys)
                tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file,validation_features,sess,x,W,b,y, keep_prob, training, emission_arr, pos_tags, model)
                pairing = (matrix, acc_val)
                multiclass.append(pairing)
                summary_statement(ii, acc_train, acc_val, tp1, fp, tp2, fn, matrix)
                pickle.dump(multiclass, open("multiclass", "wb"))
                validation_scores.append(acc_val)
                if acc_val > target_score:
                    target_score = acc_val
                    save_path = saver.save(sess, model_path + model_name)
                print "target_score: " + str(target_score)
                if genetic_algorithm == 1:
                    if acc_train < threshold_error:
                        if hidden == None:
                            population_weights.append(sess.run(W))
                            population_biases.append(sess.run(b))
                            weights = [W]
                            biases = [b]
                        elif hidden.__len__() == 2:
                            population_weights.append((sess.run(W1), sess.run(W2)))
                            population_biases.append((sess.run(b1), sess.run(b2)))
                            weights = [W1, W2]
                            biases = [b1, b2]
                        elif hidden.__len__() == 3:
                            population_weights.append((sess.run(W1), sess.run(W2), sess.run(W3)))
                            population_biases.append((sess.run(b1), sess.run(b2), sess.run(b3)))
                            weights = [W1, W2, W3]
                            biases = [b1, b2, b3]
                        fscores.append(acc_val)
                        counter -= 1
                        if counter == 0:
                            print "Starting to optimize using f-scores"
                            new_weights, new_biases = optimize_parameters.apply_op(training_file, training_features,
                                        test_file, validation_features,sess,x, weights, biases, y, population_weights,
                                        population_biases, keep_prob, training, emission_arr, pos_tags, model,
                                        total_population, fscores, saver, model_path, target_score, generations,
                                        mutation_rate, probabilistic, training_set_factor, testing_set_factor)
                            for i in range(0, weights.__len__()):
                                op1 = weights[i].assign(new_weights[i])
                                op2 = biases[i].assign(new_biases[i])
                                sess.run(op1)
                                sess.run(op2)
                            save_path = saver.save(sess, model_path + model_name)
                            population_weights = []
                            population_biases = []
                            fscores = []
                            counter = initial_population
                            break
            if ii == stop:
                tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file, validation_features, sess, x, W, b, y, keep_prob, training, emission_arr, pos_tags, model, final=1)
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
        tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file,validation_features,sess,x,W,b,y, keep_prob, training, emission_arr, pos_tags, model, 1)
        if validation_scores.__len__() > 0:
            if acc_val < np.max(validation_scores) - 0.01:
                acc_val = np.max(validation_scores)
        summary_statement("Final", acc_train, acc_val, tp1, fp, tp2, fn, matrix)
        t2 = datetime.datetime.now()
        print "Predictions complete!!!"
        print "It took " + str(t2-t1) + " to complete the evaluation"
        write_predictions(predictions, test_file)
    return scores3_train, scores3_val, scores3_precision, scores3_recall


def write_validation(path, val_score, precision, recall, resolution):
    write_file = open(path, 'w')
    max_score = np.max(val_score)
    max_index = np.argmax(val_score)
    write_file.write(str(max_score) + "\n")
    write_file.write("precision: " + str(precision[max_index]) + ", recall: " + str(recall[max_index]) + "\n")
    write_file.write("iteration: " + str(max_index) + "\n")

def load_char2vec(path):
    input_file = open(path)
    char2vec = {}
    line = input_file.readline()
    vector = []
    word = ""
    while line != "":
        k = line.split()
        start = 0
        if k.__contains__(":"):
            if k[1] == ":":
                word = k[0]
                if k[2] == "[":
                    start = 3
                else:
                    k[2] = k[2][1:]
                    start = 2
        for i in range(start, k.__len__()):
            val = k[i]
            if not val.__contains__("]"):
                vector.append(float(val))
            else:
                if val.__len__() > 1:
                    val = val[:val.__len__() - 1]
                    vector.append(float(val))
                if vector.__len__() != 50:
                    print "problem with embedding vector size"
                char2vec[word] = vector
                vector = []
        line = input_file.readline()
    input_file.close()
    return char2vec

def group_feats(features, labels, window):
    new_features = []
    new_labels = []
    for i in range(features.__len__()):
        sent_feat = features[i]
        sent_labels = labels[i]
        new_sent_feat = []
        new_sent_label = []
        for j in range(sent_feat.__len__()):
            single_feat = []
            for k in range(j - window, j + window + 1):
                if k < 0 or k >= sent_feat.__len__():
                    single_feat = np.concatenate((single_feat, np.zeros(sent_feat[0].__len__())), axis=0)
                else:
                    single_feat = np.concatenate((single_feat, sent_feat[k]),axis=0)
            new_sent_feat.append(single_feat)
            new_sent_label.append(sent_labels[j])
        new_features.append(new_sent_feat)
        new_labels.append(new_sent_label)
    print "Feature size after grouping: " + str(np.shape(new_features))
    print "Label size after grouping: " + str(np.shape(new_labels))
    return new_features, new_labels

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print "CPU count: " + str(multiprocessing.cpu_count())
    print "Starting program at " + str(t1)
    sess = tf.Session()
    if word_shape == 1:
        if seq2seq == 1:
            char2vec = char_seq2seq(PATH + "fold_" + FOLD + "_train", PATH + "fold_" + FOLD + "_test", char_seq2seq_path,
                         normalize_cap, buckets, char_seq2seq_hidden, char_seq2seq_layers, max_gradient_norm,
                         batch_size, learning_rate, learning_rate_decay_factor, use_lstm, ADAM=ADAM, stop=stop_seq2seq)
        else:
            char2vec = char_seq(PATH + "fold_" + FOLD + "_train", PATH + "fold_" + FOLD + "_test", char_seq2seq_path,
             normalize_cap, char_seq2seq_hidden, char_seq2seq_layers, max_gradient_norm, batch_size,
             learning_rate, learning_rate_decay_factor, seq2seq_decay_steps, seq2seq_staircase, cus_LSTM,
             char_drop, backward_seq2seq, orthogonal_seq2seq, stop_seq2seq, with_start, cross_val_eval, evaluate,
                                pretrained_chars)
        # else:
        #     char2vec = load_char2vec(char_file)

    else:
        char2vec = None
    predict = 1
    tf.reset_default_graph()
    new_graph = tf.Graph()  # Create an empty graph
    new_graph.as_default()
    if test == 1:
        num = 1
        splits = 1
    #index1 = WORD2VEC.find('_')
    #index2 = WORD2VEC.find('d',index1)
    #if tweet == 0:
    #    DIM = int(WORD2VEC[index1+1:index2])
    print "Loading model...."
    if fasttext_vectors == 0:
        if Google_vecs == 1:
            DIM = 300
            model = gensim.models.Word2Vec.load_word2vec_format(WORD2VEC_PATH + 'GoogleNews-vectors-negative300.bin.gz',
                                                                binary=True)
            print "GoogleNews-vectors loaded!!!"
        else:
            try:
                model = gensim.models.Word2Vec.load(WORD2VEC_PATH + WORD2VEC)
                print "Required WORD2VEC Model loaded!!!"
            except:
                model = gensim.models.Word2Vec.load(WORD2VEC_PATH + WORD2VEC2)
                print "Alternative WORD2VEC Model loaded!!!"
            #gensim.models.Word2Vec.
    else:
        print "Path: " + str(WORD2VEC_PATH + FASTTEXT)
        model = fasttext.load_model(WORD2VEC_PATH + FASTTEXT)
        print "FASTTEXT model loaded!"

    if test == 1:
        print "Extracting features from dimsum16.train"
        training_features, training_labels = features.rnn_features(PATH + "dimsum16.train", WINDOW, FEAT_WIN, POS_WINDOW,
        TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, tag_features, token_features, lemma_features, pos_features,
        sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
        tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
        remove_target, scaler, feat_preprocess, scale_data, imputer, imputing, impute_data, TOKENS_PATH + "dimsum16.train_vectors",
        LEMMAS_PATH + "dimsum16.train_vectors", uni_skip, bi_skip, char2vec)
    elif sample == 1:
        print "Extracting features from fold_test"
        training_features, training_labels = features.rnn_features(PATH + "fold_val", WINDOW, FEAT_WIN, POS_WINDOW,
        TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, tag_features, token_features, lemma_features, pos_features,
        sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
        tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
        remove_target, scaler, feat_preprocess, scale_data, imputer, imputing, impute_data, TOKENS_PATH + "fold_test_vectors",
        LEMMAS_PATH + "fold_test_vectors", uni_skip, bi_skip, char2vec)
    else:
        print "Extracting features from fold_" + FOLD + "_train"
        training_features, training_labels = features.rnn_features(PATH + "fold_" + FOLD + "_train",
        WINDOW, FEAT_WIN, POS_WINDOW, TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, tag_features,
        token_features, lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas,
        normalize, debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg,
        sentence_embedding, gen_vec, unk, unknown_token, remove_target, scaler, feat_preprocess, scale_data, imputer,
        imputing, impute_data, TOKENS_PATH + "fold_" + FOLD + "_train_vectors", LEMMAS_PATH + "fold_" + FOLD + "_train_vectors",
        uni_skip, bi_skip, char2vec)
    training_features, training_labels = group_feats(training_features, training_labels, WINDOW)
    print "Extracting validation data..."
    if test == 1:
        print "Extracting features from dimsum16.test"
        validation_features, validation_labels = features.rnn_features(PATH + "dimsum16.test", WINDOW, FEAT_WIN, POS_WINDOW,
        TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,1, ngram, tag_features, token_features, lemma_features, pos_features,
        sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
        tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
        remove_target, scaler, feat_preprocess, scale_data, imputer, imputing, impute_data, TOKENS_PATH + "dimsum16.test.blind_vectors",
        LEMMAS_PATH + "dimsum16.test.blind_vectors", uni_skip, bi_skip, char2vec)
    elif sample == 1:
        print "Extracting features from fold_test"
        validation_features, validation_labels  = features.rnn_features(PATH + "fold_test", WINDOW, FEAT_WIN, POS_WINDOW,
        TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,1, ngram, tag_features, token_features, lemma_features, pos_features,
        sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
        tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg, sentence_embedding, gen_vec, unk, unknown_token,
        remove_target, scaler, feat_preprocess, scale_data, imputer, imputing, impute_data, TOKENS_PATH + "fold_test_vectors",
        LEMMAS_PATH + "fold_test_vectors", uni_skip, bi_skip, char2vec)
    else:
        print "Extracting features from fold_" + FOLD + "_test"
        validation_features, validation_labels = features.rnn_features(PATH + "fold_" + FOLD + "_test",
        WINDOW, FEAT_WIN, POS_WINDOW, TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,1, ngram, tag_features,
        token_features, lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas,
        normalize, debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, poshmm, reg,
        sentence_embedding, gen_vec, unk, unknown_token, remove_target, scaler, feat_preprocess, scale_data, imputer,
        imputing, impute_data, TOKENS_PATH + "fold_" + FOLD + "_train_vectors",
        LEMMAS_PATH + "fold_" + FOLD + "_train_vectors", uni_skip, bi_skip, char2vec)
    validation_features, validation_labels = group_feats(validation_features, validation_labels, WINDOW)
    print "Training Neural Network..."
    if test == 1:
        train_score, val_score, precision, recall = train_NN(PATH + "dimsum16.train", training_features,training_labels,validation_features,
        validation_labels, emission_arr, pos_tags, tag_set.__len__(),cost_function, learning_rate,max_iter,predict, rho,
        batch_splits, PATH + "dimsum16.test", model, hidden)
    elif sample == 1:
        train_score, val_score, precision, recall = train_NN(PATH + "fold_val", training_features, training_labels, validation_features,
        validation_labels, emission_arr, pos_tags, tag_set.__len__(), cost_function, learning_rate, max_iter, predict, rho,
        batch_splits, PATH + "fold_test", model, hidden)
    else:
        train_score, val_score, precision, recall = train_NN(PATH + "fold_" + FOLD + "_train", training_features,training_labels,validation_features,
        validation_labels, emission_arr, pos_tags, tag_set.__len__(),cost_function,learning_rate,max_iter,predict, rho,
        batch_splits, PATH + "fold_" + FOLD + "_test", model, hidden)
    write_validation(PATH + "fold_" + FOLD + "_val", val_score, precision, recall, resolution)
    t2 = datetime.datetime.now()
    print "The program ended at " + str(t2)
    print "It took " + str(t2 - t1) + " to finish the evaluation"

