
import tensorflow as tf
import gensim
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import ujson
import extract_mwes
import sys
import features
import skipthoughts
import operator
import error_analysis

WINDOW = 1
POS_WINDOW = 2
TAG_WINDOW = 0
MWE_WINDOW = 2
DIM = 100
embedding_window = 5
tag_features = 0
token_features = 1
lemma_features = 1
pos_features = 1
context = 1
binary_transitions = 0
in_mwe = 1
tag_distribution = 0
ngram = 3
rho = 50
cost_function = 'sce' #sce
#hidden = [(200,'tanh'),(50,'tanh'),(6,'soft')] # 400 relu, 100 tanh = 61.2%; 400 relu, 100 relu = <60%
hidden = [(440,'tanh'),(6,'soft')]
#hidden = None
test = 0
max_iter = 600000
reg = 1e10
sentence_embedding = 0
learning_rate = 1e-2
gen_vec = 0
unk = 0
remove_target = 0
normalize = 0
sample = 0
batch_size = 0
uni_skip = 0
bi_skip = 0
Google_vecs = 0


def parse_input(x):
    if x.__contains__(','):
        k = x.split(',')
        y = ''.join(k)
        return tuple(y)
    return int(x)


WINDOW = parse_input(sys.argv[1])
POS_WINDOW = parse_input(sys.argv[2])
TAG_WINDOW = parse_input(sys.argv[3])
MWE_WINDOW = parse_input(sys.argv[4])
DIM = int(sys.argv[5])
embedding_window = int(sys.argv[6]) #4
tag_features = int(sys.argv[7])
lemma_features = int(sys.argv[8])
pos_features = int(sys.argv[9])
context = int(sys.argv[10])
#binary_transitions = int(sys.argv[8])
in_mwe = int(sys.argv[11])
#tag_distribution = int(sys.argv[10])
ngram = int(sys.argv[12])
rho = int(sys.argv[13])
cost_function = sys.argv[14] #sce
hidden = [(int(sys.argv[15]),'tanh'),(6,'soft')]
test = int(sys.argv[16])
max_iter = int(sys.argv[17])
sentence_embedding = 0
learning_rate = 1e-2
#gen_vec = int(sys.argv[17])
remove_target = int(sys.argv[18])
normalize = int(sys.argv[19])
con_win = int(sys.argv[20]) # 3 = 65.7%, 5 = 66.4%
avg_tokens = 1
avg_lemmas = 1
WORD2VEC = "Word2Vec_" + str(DIM) + "d_" + str(embedding_window) + "window_15mincount_5nsampling_skipgram.bin"
WORD2VEC2 = "wikiEmbeddingsStanfordTokenizedSkipGram" + str(embedding_window) + "-dimension" + str(DIM) + "vectors.bin"
WORD2VEC_PATH = "/home/waseem/models/"
PATH = "/home/waseem/Downloads/dimsum-data-1.5/"
TOKENS_PATH = PATH + 'tokens/'
LEMMAS_PATH = PATH + 'lemmas/'
splits = 5
num = 5
repetitions = 1
 #400
resolution = 100

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
#binary_transition == 1, sig = 55.4%
#hidden = [(205,'relu'),(35,'relu'),(6,'soft')] #tanh, tanh = 56.4%
#hidden = None
pos_tags = []
emission_arr = []


start_tag = 'start'
end_tag = 'end'
tag_set = ['B', 'I', 'O', 'b', 'i', 'o']
#tag_set = ['B', 'I', 'O']


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev= 0.1)
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

def check_sequence(sequence):
    inside = 0
    for element in sequence:
        if element[4] == 'B':
            inside = 1
        elif element[4] == 'O':
            if inside == 1:
                inside = 0
            else:
                continue
        if inside == 0 and (element[4] != 'O' or element[4] != 'B'):
            return False
    return True


def amend_sequence(sequence):
    inside = 0
    index1 = sequence.__len__()
    index2 = sequence.__len__()
    for i in range(sequence.__len__()-1,-1,-1):
        if i == sequence.__len__() - 1 and sequence[i][4] != 'I':
            sequence[i][4] = 'O'
        elif i == 0 and sequence[i][4] == 'I':
            sequence[i][4] = 'B'
            inside = 0
            sequence[index1][5] = str(i+1)
        elif inside == 1:
            if sequence[i][4] == 'O':
                sequence[i][4] = 'B'
            if sequence[i][4] == 'B':
                inside = 0
                sequence[index1][5] = str(i+1)
            if sequence[i][4] == 'I':
                sequence[index1][5] = str(i+1)
                index1 = i
            if sequence[i][4] == 'i':
                inside = 2
                index2 = i
            if sequence[i][4] == 'b':
                sequence[i][4] = 'o'
        elif inside == 2:
            if sequence[i][4] != 'i':
                sequence[i][4] = 'b'
                sequence[index2][5] = str(i+1)
                inside = 1
            else:
                sequence[index2][5] = str(i+1)
                index2 = i
        elif inside == 0:
            if sequence[i][4] == 'I':
                inside = 1
                index1 = i
            else:
                sequence[i][4] = 'O'
        if sequence[i][4] != 'I' or sequence[i][4] != 'i':
            sequence[i][5] = '0'
        sequence[i][7] = ''
    return sequence


def write_predictions(classification, test_file):
    reference = open(test_file)
    new_file = open(test_file + '.pred','w')
    rline = reference.readline()
    i = 0
    sequence = []
    j = 1
    index1 = -1
    index2 = -1
    while rline != '':
        if rline != '\n':
            k = rline.split('\t')
            current_tag = classification[i]
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
            i += 1
            j += 1
            sequence.append(k)
        else:
            for element in sequence:
                new_file.write('\t'.join(element))
            new_file.write('\n')
            sequence = []
            j = 1
        rline = reference.readline()
    reference.close()
    new_file.close()


def plot_scores(index,train,val,fig_num,leg,flag):
    plt.figure(fig_num)
    ax = plt.subplot(111)
    #plt.ion()
    ax.plot(index,train,'--',index,val)
    plt.ylabel('Accuracy')
    box = ax.get_position()
    if flag == 0:
        ax.set_position([box.x0, box.y0 + box.height*0.18,box.width, box.height*0.9])
        flag = 1
    ax.legend(leg,loc= 'lower center', fancybox= True, bbox_to_anchor=(0.48,-0.35),ncol=4)
    plt.show(block= False)
    #plt.draw()
    plt.pause(0.001)
    #plt.show(block=False)
    return flag


def plot_error_bars(index,train,train_std,val,val_std,fig_num,leg,flag):
    plt.figure(fig_num)
    ax = plt.subplot(111)
    #plt.ion()
    ax.plot(index,train,'--',index,val)
    plt.ylabel('Accuracy')
    box = ax.get_position()
    if flag == 0:
        ax.set_position([box.x0, box.y0 + box.height*0.18,box.width, box.height*0.9])
        flag = 1
    ax.legend(leg,loc= 'lower center', fancybox= True, bbox_to_anchor=(0.48,-0.35),ncol=4)
    plt.errorbar(index, train, yerr= train_std,linestyle="None")
    plt.errorbar(index, val, yerr= val_std,linestyle="None")
    plt.pause(0.001)
    plt.show(block=False)
    return flag


def get_stats(classification):
    dict = {}
    for val in classification:
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
        new_sequence.append(best_tuple[i])
        i -= 1
    tag = best_tuple[i+1]
    i = sequence.__len__()
    while tag != start_tag:
        #print "i = " + str(i) + ", max_tuple = " + str(max_tuple)
        #print bp
        best_tuple = bp[(i,best_tuple)]
        tag = best_tuple[0]
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

def get_pred(test_file,validation_features,sess,x,W,b,y,model = None,final = 0):
    predictions = []
    gold_sequence = []
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
    while line != '':
        if line != '\n':
            k = line.split('\t')
            token = k[1]
            model_token = "N/A"
            if model.__contains__(token):
                model_token = token
            elif model.__contains__(token.lower()):
                model_token = token.lower()
            token_arr.append(model_token)
            lemma = k[2]
            model_lemma = "N/A"
            if model.__contains__(lemma):
                model_lemma = lemma
            elif model.__contains__(lemma.lower()):
                model_lemma = lemma.lower()
            lemma_arr.append(model_lemma)
            pos = k[3]
            pos_arr.append(pos)
            tag = k[4]
            line_arr = [k[0], k[1], model_token, k[2], model_lemma, k[3], k[4]]
            line_by_line.append(line_arr)
            pred = sess.run(y,feed_dict={x:[validation_features[j]]})
            cl = np.zeros(pred[0].__len__())
            if tag_features == 1:
                max_index = np.argmax(pred[0])
                cl[max_index] = 1
                cl = pred[0]
            #cl,inside = amend_pred(pred[0],inside)
            #print "Classification: " + str(cl)
            #predictions.append(cl.index(1))
            #print "The prediction is " + str(cl.index(1))
            sequence.append(pred[0])
            gold_sequence.append(tag_set[validation_labels[j].index(1)])
            sum_tags = 0
            start = DIM
            if lemma_features == 1:
                start += DIM
            if pos_features == 1:
                start += pos_tags[0].__len__()
            start += tag_set.__len__()
            if j != validation_features.__len__() - 1:
                for i in range(1,int(WINDOW)+1):
                    sum_tags += np.sum(validation_features[j+1][start*i - pos_tags[0].__len__() - tag_set.__len__():start*i - tag_set.__len__()])
                #print "sum_tags = " + str(sum_tags)
                if tag_features == 1 and sum_tags != 0:
                    end = validation_features[j+1].__len__()
                    if context == 1:
                        end -= (factor*(int(WINDOW)/2 + 1))
                        #print "The factor is " + str(factor)
                    threshold = 0
                    #if context == 1:
                    while end > threshold:
                        validation_features[j+1][end-tag_set.__len__(): end] = cl
                        cl = validation_features[j][end-tag_set.__len__():end]
                        end -= DIM
                        if lemma_features == 1:
                            end = end - DIM
                        if pos_features == 1:
                            end = end - pos_tags[0].__len__()
                        if tag_features == 1:
                            end = end - tag_set.__len__()
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
            max_predictions = Viterbi(sequence, tag_set, emission_arr[0])
            #print "max_predictions: " + str(max_predictions)
            predictions += max_predictions
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
            gold_sequence = []
            line_by_line = []
        line = keep_track.readline()
    if final == 1:
        analyze.close()
    keep_track.close()
    precision = div(tp1,tp1+fp)
    recall = div(tp2,tp2 + fn)
    fscore = get_fscore(precision,recall)
    acc_val = fscore
    train_criteria = get_stats(predictions)
    #error_analysis.error_stats([token_arr, lemma_arr, pos_arr],[10,10,10],validation_labels, predictions, tag_set)
    #error_analysis.stats_by_percentage([token_arr, lemma_arr, pos_arr],[10,10,10],[10,10,10],validation_labels, predictions, tag_set)
    word_features = ['cap','anycap','upper','alpha','num','nonchar','http']
    error_analysis.word_stats_by_percentage([token_arr, lemma_arr],[10,10],[10,10],word_features,validation_labels, predictions, tag_set)
    if predictions.__len__() != validation_labels.__len__():
        print "Predictions and labels are not equal!!!"
    print train_criteria
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


def train_NN(training_features,training_labels, validation_features, validation_labels, classes, cost_fun, alpha, stop, predict, rho, batch_size, test_file, model = None, hidden = None):
    acc_train = 0
    acc_val = 0
    input_size = training_features[0].__len__()
    x = tf.placeholder(tf.float32, [None, input_size])
    #x_input = [x]
    #Holds variables, 'None' indicates unknown length
    W = []
    b = []
    y_ = tf.placeholder(tf.float32, [None, classes])
    #keep_prob = tf.placeholder(tf.float32)
    if hidden == None:
        l1 = classes
        W = weight_variable([input_size,l1])
        #This is how a variable is declared, the model parameters should be declared as variables
        b = bias_variable([l1])
        #y1 = tf.nn.relu(tf.matmul(x,W) + b)
        intermediate = tf.matmul(x,W) + b
        y1 = tf.nn.softmax(intermediate)
        y = y1
    elif hidden.__len__() == 2:
        l1 = hidden[0][0]
        W1 = weight_variable([input_size,l1])
        #This is how a variable is declared, the model parameters should be declared as variables
        b1 = bias_variable([l1])
        activation = hidden[0][1]
        if activation == 'tanh':
            y1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
        elif activation == 'sig':
            y1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
        elif activation == 'relu':
            y1 = tf.nn.relu(tf.matmul(x,W1) + b1)
        elif activation == 'soft':
            y1 = tf.nn.softmax(tf.matmul(x,W1) + b1)
        #h_fc1_drop = tf.nn.dropout(y1, keep_prob)
        l2 = hidden[1][0]
        W2 = weight_variable([l1,l2])
        #This is how a variable is declared, the model parameters should be declared as variables
        b2 = bias_variable([l2])
        activation = hidden[1][1]
        intermediate = tf.matmul(y1,W2) + b2
        if activation == 'tanh':
            y2 = tf.nn.tanh(intermediate)
        elif activation == 'sig':
            y2 = tf.nn.sigmoid(intermediate)
        elif activation == 'relu':
            y2 = tf.nn.relu(intermediate)
        elif activation == 'soft':
            y2 = tf.nn.softmax(intermediate)
        y = y2
    elif hidden.__len__() == 3:
        l1 = hidden[0][0]
        W1 = weight_variable([input_size,l1])
        #This is how a variable is declared, the model parameters should be declared as variables
        b1 = bias_variable([l1])
        activation = hidden[0][1]
        if activation == 'tanh':
            y1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
        elif activation == 'sig':
            y1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
        elif activation == 'relu':
            y1 = tf.nn.relu(tf.matmul(x,W1) + b1)
        elif activation == 'soft':
            y1 = tf.nn.softmax(tf.matmul(x,W1) + b1)
        l2 = hidden[1][0]
        W2 = weight_variable([l1,l2])
        #This is how a variable is declared, the model parameters should be declared as variables
        b2 = bias_variable([l2])
        activation = hidden[1][1]
        intermediate1 = tf.matmul(y1,W2) + b2
        if activation == 'tanh':
            y2 = tf.nn.tanh(intermediate1)
        elif activation == 'sig':
            y2 = tf.nn.sigmoid(intermediate1)
        elif activation == 'relu':
            y2 = tf.nn.relu(intermediate1)
        elif activation == 'soft':
            y2 = tf.nn.softmax(intermediate1)
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
    # function to minimize, the [1] indicates that the summation is done on the 2nd dimension
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
    #train_step = tf.train.AdadeltaOptimizer(alpha).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(alpha).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    scores3_train = np.zeros(max_iter)
    scores3_val = np.zeros(max_iter)
    validation_scores = []
    target_score = 0
    acc_val = 0
    if pretrained == 0:
        ii = 0
        while ii < stop:
            #Training for 1000 iterations
            if batch_size > 0:
                batch = random.randint(0,training_features.__len__() - batch_size - 1)
                batch_xs, batch_ys = training_features[batch: batch + batch_size], training_labels[batch: batch + batch_size]
            else:
                batch_xs, batch_ys = training_features, training_labels
            assert training_features.__len__() == training_labels.__len__()
            reference = training_features[0].__len__()
            #print reference
            for iii in range(1,training_features.__len__()):
                if reference != training_features[iii].__len__():
                    print "element " + str(iii) + " is not equal"
                    #print str(training_features[iii]) + ", the reference is " + str(training_features[0])
                    print str(training_features[iii].__len__()) + " instead of " + str(reference)
                    #display_features(training_features[iii])
            #A batch of 100 random data points is chosen from the training set
            #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            # if (ii+1)%1000 == 0:
            #     with open(PATH + "weights_" + str(ii),'w') as f:
            #         ujson.dump(W,f)
            if ii%resolution == 0:
                #correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
                #Arg_max returns the index of the highest entry along some axis; tf.equal returns boolean data types
                #accuracy_validation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #Find the mean of the boolean predictions
                #acc_train = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                acc_train = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
                tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file,validation_features,sess,x,W,b,y, model)
                summary_statement(ii, acc_train, acc_val, tp1, fp, tp2, fn, matrix)
                validation_scores.append(acc_val)
                target_score = np.max(validation_scores)
            if ii == stop - 1:
                tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file, validation_features, sess, x, W, b, y, model)
                if acc_val < target_score - 0.02:
                    stop += 10
            if ii < scores3_train.__len__():
                scores3_train[ii] += acc_train*100
                scores3_val[ii] += acc_val*100
            ii += 1
    if predict == 1:
        if pretrained == 1:
            with open(PATH + pretrained_file) as f:
                feats = ujson.load(f)
        t1 = datetime.datetime.now()
        print "Starting predictions..."
        tp1, fp, tp2, fn, acc_val, predictions, matrix = get_pred(test_file,validation_features,sess,x,W,b,y, model, 1)
        if acc_val < np.max(validation_scores) - 2:
            acc_val = np.max(validation_scores)
        summary_statement("Final", acc_train, acc_val, tp1, fp, tp2, fn, matrix)
        t2 = datetime.datetime.now()
        print "Predictions complete!!!"
        print "It took " + str(t2-t1) + " to complete the evaluation"
        write_predictions(predictions, test_file)
    return scores3_train,scores3_val



if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print "Starting program at " + str(t1)
    predict = 1
    if test == 1:
        predict = 1
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
                training_features, training_labels = features.nn_features(PATH + "dimsum16.train",WINDOW, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, binary_transitions, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, reg, sentence_embedding, gen_vec, unk,
                remove_target, TOKENS_PATH + "dimsum16.train_vectors", LEMMAS_PATH + "dimsum16.train_vectors",
                uni_skip, bi_skip)
            elif sample == 1:
                print "Extracting features from fold_test"
                training_features, training_labels = features.nn_features(PATH + "fold_test", WINDOW, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, binary_transitions, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, reg, sentence_embedding, gen_vec, unk,
                remove_target, TOKENS_PATH + "fold_test_vectors", LEMMAS_PATH + "fold_test_vectors", uni_skip, bi_skip)
            else:
                print "Extracting features from fold_" + str(i+1) + str(j+1) + "_train"
                training_features, training_labels = features.nn_features(PATH + "fold_" + str(i+1) + str(j+1) + "_train",
                WINDOW, POS_WINDOW, TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, binary_transitions, tag_features,
                token_features, lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas,
                normalize, debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, reg,
                sentence_embedding, gen_vec, unk, remove_target, TOKENS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors",
                LEMMAS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors", uni_skip, bi_skip)
            print "Extracting validation data..."
            if test == 1:
                print "Extracting features from dimsum16.test.blind"
                validation_features, validation_labels = features.nn_features(PATH + "dimsum16.test.blind", WINDOW, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, binary_transitions, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, reg, sentence_embedding, gen_vec, unk,
                remove_target, TOKENS_PATH + "dimsum16.test.blind_vectors", LEMMAS_PATH + "dimsum16.test.blind_vectors",
                uni_skip, bi_skip)
            elif sample == 1:
                print "Extracting features from fold_test"
                validation_features, validation_labels  = features.nn_features(PATH + "fold_test", WINDOW, POS_WINDOW,
                TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, binary_transitions, tag_features, token_features, lemma_features, pos_features,
                sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe,
                tag_distribution, separate_lexicons,  pos_tags, emission_arr, reg, sentence_embedding, gen_vec, unk,
                remove_target, TOKENS_PATH + "fold_test_vectors", LEMMAS_PATH + "fold_test_vectors", uni_skip, bi_skip)
            else:
                print "Extracting features from fold_" + str(i + 1) + str(j + 1) + "_test"
                validation_features, validation_labels = features.nn_features(PATH + "fold_" + str(i+1) + str(j+1) + "_test",
                WINDOW, POS_WINDOW, TAG_WINDOW, MWE_WINDOW, DIM, tag_set, model,0, ngram, binary_transitions, tag_features,
                token_features, lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas,
                normalize, debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, reg,
                sentence_embedding, gen_vec, unk, remove_target, TOKENS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors",
                LEMMAS_PATH + "fold_" + str(i+1) + str(j+1) + "_train_vectors", uni_skip, bi_skip)
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
                    train_score, val_score = train_NN(training_features,training_labels,validation_features,
                    validation_labels,tag_set.__len__(),cost_function, learning_rate,max_iter,predict, rho,
                    batch_size, PATH + "dimsum16.test.blind", model, hidden)
                elif sample == 1:
                    train_score, val_score = train_NN(training_features, training_labels, validation_features,
                    validation_labels, tag_set.__len__(), cost_function, learning_rate, max_iter, predict, rho,
                    batch_size, PATH + "fold_test", model, hidden)
                else:
                    train_score, val_score = train_NN(training_features,training_labels,validation_features,
                    validation_labels,tag_set.__len__(),cost_function,learning_rate,max_iter,predict, rho,
                    batch_size, PATH + "fold_" + str(i+1) + str(j+1) + "_test", model, hidden)
                scores3_train[rep] += train_score
                scores3_val[rep] += val_score
                #flag3 = plot_scores(index,scores3_train[rep],scores3_val[rep],100*i + 10*j + 3,leg3,flag3)
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

