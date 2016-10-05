MWE_LEX = "/home/waseem/Downloads/pysupersensetagger-2.0/mwelex/"

import numpy as np
import extract_mwes
import string


start_tag = 'start'
end_tag = 'end'
tag_set = ['B', 'I', 'O', 'b', 'i', 'o']

def get_pos_hmm(path, ngram, alpha):
    pos_tags = []
    training_file = open(path)
    line = training_file.readline()
    emission_count = {}
    counts = {}
    tag_tuple = ()
    for i in range(0,ngram):
        tag_tuple += (start_tag,)
    while line != '':
        if line != '\n':
            k = line.split('\t')
            pos = k[3]
            tag = k[4]
            if not pos_tags.__contains__(pos):
                pos_tags.append(pos)
            tag_tuple = tag_tuple[1:] + (tag,)
            if emission_count.has_key(tag_tuple):
                emission_count[tag_tuple] += 1
            else:
                emission_count[tag_tuple] = 1
        else:
            tag_tuple = tag_tuple[1:] + (end_tag,)
            tokens = []
            lemmas = []
            if emission_count.has_key(tag_tuple):
                emission_count[tag_tuple] += 1
            else:
                emission_count[tag_tuple] = 1
            tag_tuple = ()
            for i in range(0,ngram):
                tag_tuple += (start_tag,)
        line = training_file.readline()
    training_file.close()
    #print "emission_count = " + str(emission_count)
    emission_array = [{}, {}]
    for i in range(0,emission_array.__len__()):
        for tag_tuple in emission_count:
            temp = emission_array[i]
            if not temp.__contains__(tag_tuple[:i+1]):
                if i > 0:
                    temp[tag_tuple] = (emission_count[tag_tuple] + alpha)*1.0/(emission_array[i-1][tag_tuple[:i]] + alpha*emission_count.__len__())
                else:
                    temp[tag_tuple[:i+1]] = emission_count[tag_tuple]
            else:
                if i > 0:
                    temp[tag_tuple] += (emission_count[tag_tuple])*1.0/(emission_array[i-1][tag_tuple[:i]] + alpha*emission_count.__len__())
                else:
                    temp[tag_tuple[:i+1]] += emission_count[tag_tuple]
    #print emission_array[0]
    print emission_array[1]
    return pos_tags, emission_array[1]


def get_pos_poshmm(path, ngram, alpha):
    pos_tags = []
    training_file = open(path)
    line = training_file.readline()
    emission_count = {}
    counts = {}
    tag_tuple = ()
    for i in range(0,ngram):
        tag_pos = (start_tag,)
        tag_tuple += (tag_pos,)
    while line != '':
        if line != '\n':
            k = line.split('\t')
            pos = k[3]
            tag = k[4]
            tag_pos = (tag, pos)
            if not pos_tags.__contains__(pos):
                pos_tags.append(pos)
            tag_tuple = tag_tuple[1:] + (tag_pos,)
            if emission_count.has_key(tag_tuple):
                emission_count[tag_tuple] += 1
            else:
                emission_count[tag_tuple] = 1
        else:
            tag_tuple = tag_tuple[1:] + ((end_tag,),)
            tokens = []
            lemmas = []
            if emission_count.has_key(tag_tuple):
                emission_count[tag_tuple] += 1
            else:
                emission_count[tag_tuple] = 1
            tag_tuple = ()
            for i in range(0,ngram):
                tag_pos = (start_tag,)
                tag_tuple += (tag_pos,)
        line = training_file.readline()
    training_file.close()
    #print "emission_count = " + str(emission_count)
    emission_array = [{}, {}]
    for i in range(0,emission_array.__len__()):
        for tag_tuple in emission_count:
            temp = emission_array[i]
            if not temp.__contains__(tag_tuple[:i+1]):
                if i > 0:
                    temp[tag_tuple] = (emission_count[tag_tuple] + alpha)*1.0/(emission_array[i-1][tag_tuple[:i]] + alpha*emission_count.__len__())
                else:
                    temp[tag_tuple[:i+1]] = emission_count[tag_tuple]
            else:
                if i > 0:
                    temp[tag_tuple] += (emission_count[tag_tuple])*1.0/(emission_array[i-1][tag_tuple[:i]] + alpha*emission_count.__len__())
                else:
                    temp[tag_tuple[:i+1]] += emission_count[tag_tuple]
    #print emission_array[0]
    print emission_array[1]
    return pos_tags, emission_array[1]

def tag_feat(tag,tag_set):
    features = []
    for ts in tag_set:
        if ts == tag:
            features.append(1)
        else:
            features.append(0)
    return features


def extractLexiconCandidates(sent, mwe_lex):
    if mwe_lex is not None:
        extract_mwes.load_lexicons(mwe_lex)
    # if program_args.lex is not None:
    #     mwe_lexicons.load_lexicons(program_args.lex)
    #if program_args.clist is not None:
        #mwe_lexicons.load_combined_lexicon('mwetoolkit YelpAcademic', [f for f in program_args.clist if f.name.startswith('mwetk.yelpAcademic')], is_list=True)
        #mwe_lexicons.load_lexicons([f for f in program_args.clist if not f.name.startswith('mwetk.yelpAcademic')], is_list=True)
        #mwe_lexicons.load_lexicons(program_args.clist, is_list=True)
    '''
    For each lexicon and collocation list, compute the shortest-path lexical segmentation
    of the sentence under that lexicon.
    Return a list of MWE membership information tuples for each token
    according to that segmentation.
    '''
    #assert mwe_lexicons._lexicons   # actually, depends on whether any --lex args are present...
    sentence_lemmas = [t for t in sent]
    return ({lexiconname: lex.shortest_path_decoding(sentence_lemmas, max_gap_length=2)[2]
            for lexiconname,lex in extract_mwes._lexicons.items()},
            {listname: lex.shortest_path_decoding(sentence_lemmas, max_gap_length=2)[2]
            for listname,lex in extract_mwes._lists.items()})

def get_vector(model,word_arr, index, con_win, word_dim, normalize, one_way, gen_vec, remove_word, unk, unk_token):
    resultant = 0
    temp = np.zeros(word_dim)
    norm = 0
    if one_way == 0:
        min_index = index - con_win
        min_index = max(min_index,0)
        max_index = index + con_win + 1
        max_index = min(max_index, word_arr.__len__())
        for i in range(min_index, max_index):
            if i != remove_word:
                #temp1 = np.zeros(word_dim)
                temp1 = append_features(model, word_arr[i], word_dim, gen_vec, unk, unk_token)
                #if model.__contains__(word_arr[i]):
                #    temp1 = model[word_arr[i]]
                temp += temp1
                if np.sum(temp1) != 0:
                    norm += 1
        if normalize == 1 and np.sum(temp) > 0 and gen_vec == 0:
            temp /= norm
        else:
            temp /= con_win
        resultant = temp
    if one_way == 1:
        if con_win + index > word_arr.__len__():
            con_win = word_arr.__len__() - index
        for i in range(index, index + con_win):
            if i != remove_word:
                if i != remove_word:
                    temp1 = append_features(model, word_arr[i], word_dim, gen_vec, unk, unk_token)
                    temp += temp1
                    if np.sum(temp1) != 0:
                        norm += 1
        if normalize == 1 and gen_vec == 0:
            resultant = temp/norm
        else:
            resultant = temp/con_win
    return resultant

def get_stats(classification):
    dict = {}
    for val in classification:
        index = tag_set.index(val)
        if dict.__contains__(tag_set[index]):
            dict[tag_set[index]] += 1
        else:
            dict[tag_set[index]] = 1
    return dict


def get_general_vec(model, word_dim):
    vector = np.zeros(word_dim)
    num = 0
    for word in model.vocab:
        #print "word: " + str(word)
        vector += model[word]
        num += 1
    general_vec = vector/num
    return general_vec

def read_vector(line):
    vector = []
    k = line.split('\t')
    k.pop()
    #print "The size of the line is " + str(k.__len__())
    #print 'The last element is ' + str(k[k.__len__() - 1])
    for val in k:
        vector.append(float(val))
    return vector

def get_skipthoughts(path = None):
    skipthoughts_vectors = []
    if path != None:
        skipthoughts_file = open(path)
        skipthoughts_line = skipthoughts_file.readline()
        while skipthoughts_line != '':
            temp_vec = read_vector(skipthoughts_line)
            skipthoughts_vectors.append(temp_vec)
            skipthoughts_line = skipthoughts_file.readline()
    return skipthoughts_vectors


def preprocess(word):
    if word.__contains__('#'):
        anomaly = word.index('#')
        word = word[:anomaly] + word[anomaly + 1:]
    if word.__contains__('@'):
        anomaly = word.index('@')
        word = word[:anomaly] + word[anomaly + 1:]
    if word.__contains__('http'):
        word = 'URL'
    is_a_num = 1
    for char in word:
        if not string.digits.__contains__(char):
            is_a_num = 0
    if is_a_num == 1:
        word = 'NUMBER'
    return word


def append_features(model, word, word_dim, gen_vec, unk, unknown_token):
    feature = []
    unk_array = []
    vector = np.zeros(word_dim)
    new_word = preprocess(word)
    if new_word != 'NUMBER' and new_word != 'URL': # and not string.digits.__contains__(word):
        if model.__contains__(word):
            vector = model[word]
        elif model.__contains__(word.lower()):
            vector = model[word.lower()]
        else:
            # print "feature size was " + str(features.__len__())
            if unk == 1:
                vector = model[unknown_token]
            elif gen_vec == 1:
                vector = general_vec
        # print "feature size is " + str(features.__len__())
    feature = np.concatenate((feature, vector), axis=0)
    #print "feature length: " + str(features.__len__())
    if unk == 1:
        feature = np.concatenate((feature, unk_array), axis=0)
    #print "feature length2: " + str(features.__len__())
    return feature


def context2vec(avg_tokens, remove, model, word_arr, word_dim, normalize, sentence_embedding, gen_vec, index, uni_skip,
                bi_skip, skipthoughts_word_vectors, sentence_counter, con_win, unk, unk_token):
    word_vec = []
    if avg_tokens == 1:
        if remove == 1:
            word_vec = get_vector(model, word_arr, index, con_win, word_dim, normalize, sentence_embedding,
                                   gen_vec, index, unk, unk_token)
        else:
            # print sentence_counter
            if uni_skip == 1:
                word_vec += skipthoughts_word_vectors[sentence_counter][0:2400]
            if bi_skip == 1:
                word_vec += skipthoughts_word_vectors[sentence_counter][2401:4800]
            if uni_skip == 0 and bi_skip == 0:
                word_vec = get_vector(model, word_arr, index, con_win, word_dim, normalize, sentence_embedding,
                                       gen_vec, -1, unk, unk_token)
    return word_vec


def extra_word_features(word_arr, j):
    feature = []
    zeros = False
    if j < 0 or j >= word_arr.__len__():
        zeros = True
        j = 0
    #In_quotes (split into single quotes vs double quotes?)

    beginning = 0
    if j == 0:
        beginning = 1

    feature.append(beginning)

    end = 0
    if j == word_arr.__len__() - 1:
        end = 1

    feature.append(end)

    double_quote_count = 0
    single_quote_count = 0
    double_quote_feat = 0
    quote_feat = 0
    single_quote_feat = 0
    for i in range(0,j):
        if word_arr[i] == '"':
            double_quote_count += 1
        if word_arr[i] == '':
            single_quote_count += 1
    if double_quote_count == 1:
        for i in range(j + 1, word_arr.__len__()):
            if word_arr[i] == '"':
                quote_feat = 1
                double_quote_feat = 1
                break
    if single_quote_count == 1:
        for i in range(j + 1, word_arr.__len__()):
            if word_arr[i] == '':
                quote_feat = 1
                single_quote_feat = 1
                break
    feature.append(quote_feat)
    feature.append(single_quote_feat)
    feature.append(double_quote_feat)

    #All Capital
    all_caps = 1
    for char in word_arr[j]:
        if not string.ascii_uppercase.__contains__(char) and char != '@' and char != '#':
            all_caps = 0
            break
    feature.append(all_caps)

    #First_letter_capital
    capitalize = 0 #Maybe turn this feature on regardless of whether all_caps is on or not
    if all_caps == 1:
        if string.ascii_uppercase.__contains__(word_arr[j][0]):
            capitalize = 1
    feature.append(capitalize)

    # Is a number
    num = 1
    for char in word_arr[j]:
        if not string.digits.__contains__(char):
            num = 0
    feature.append(num)

    #Contains a number
    any_num = 0
    if num == 0:
        for char in word_arr[j]:
            if string.digits.__contains__(char):
                any_num = 1
    feature.append(any_num)

    #Remove # and @ when obtaining word vectors

    # Contains #
    hashtag = 0
    if word_arr[j].__contains__('#'):
        hashtag = 1
    feature.append(hashtag)

    # Contains @
    at_symbol = 0
    if word_arr[j].__contains__('@'):
        at_symbol = 1
    feature.append(at_symbol)

    #Is a hyphen

    #IS a URL (contains http or is called URL)
    url_feat = 0
    if word_arr[j] == 'URL' or word_arr[j].__contains__('http'):
        url_feat = 1
    feature.append(url_feat)

    #different window sizes for these extra features?

    # Is a non-alphanumeric
    non_alpha = 0
    if string.punctuation.__contains__(word_arr[j]):
        non_alpha = 1
    feature.append(non_alpha)

    #contains non-alphanumeric
    any_punct = 0
    if non_alpha == 0 and url_feat == 0:
        for char in word_arr[j]:
            if string.punctuation.__contains__(char) and char != '#' and char != '@':
                any_punct = 1
    feature.append(any_punct)

    all_punct = 1
    if non_alpha == 0:
        for char in word_arr[j]:
            if not string.punctuation.__contains__(char):
                all_punct = 0
    feature.append(all_punct)

    if zeros == True:
        return np.zeros(feature.__len__())

    return feature



def word_feature(word_arr, model, window, feat_window, i, context, word_dim, gen_vec, unk, unknown_token):
    feature = []
    if context == 1:
        try:
            left_window = window[0]
            right_window = window[1]
        except:
            left_window = window
            right_window = window
        s = i - left_window
        e = i + right_window + 1
        start = max(s,0)
        end = min(e, word_arr.__len__())
        for j in range(s, e):
            if j < start or j >= end:
                vector = np.zeros(word_dim)
                feature = np.concatenate((feature, vector), axis=0)
                vector = extra_word_features(word_arr, j)
                feature = np.concatenate((feature, vector), axis=0)
                #print "vector: " + str(np.zeros(word_dim))
            else:
                vector = append_features(model, word_arr[j], word_dim, gen_vec, unk, unknown_token)
                #print "vector: " + str(vector)
                feature = np.concatenate((feature, vector), axis=0)
                vector = extra_word_features(word_arr, j)
                feature = np.concatenate((feature, vector), axis=0)
        try:
            left_window = feat_window[0]
            right_window = feat_window[1]
        except:
            left_window = feat_window
            right_window = feat_window
        s = i - left_window
        e = i + right_window + 1
        for j in range(s, e):
            some_var = 0
            #vector = extra_word_features(word_arr, j)
            #feature = np.concatenate((feature, vector), axis=0)
    else:
        start = max(i, 0)
        end = min(i + window + 1, word_arr.__len__())
        for j in range(i, i + context + 1):
            if j < start or j >= end:
                feature = np.concatenate((feature, np.zeros(word_dim)), axis=0)
            else:
                vector = append_features(model, word_arr[j], word_dim, gen_vec, unk, unknown_token)
                feature = np.concatenate((feature, vector), axis=0)
    return feature


def sub_context(word_arr, model, window, i, word_dim, gen_vec, unk, unknown_token):
    try:
        left_window = window[0]
        right_window = window[1]
    except:
        left_window = window
        right_window = window
    s = i - left_window
    e = i + right_window + 1
    start = max(s,0)
    end = min(e, word_arr.__len__())
    avg_vec = np.zeros(word_dim)
    for j in range(start, end):
        if j != i:
            avg_vec = np.add(avg_vec,append_features(model, word_arr[j], word_dim, gen_vec, unk, unknown_token))
    avg_vec /= (end - start)
    feature = np.subtract(append_features(model, word_arr[i], word_dim, gen_vec, unk, unknown_token),avg_vec)
    return feature


def window_feature(array, array_set, window, i, context):
    feature = []
    #print "array: " + str(array)
    #print "array_set: " + str(array_set)
    if context == 1:
        try:
            left_window = window[0]
            right_window = window[1]
        except:
            left_window = window
            right_window = window
        s = i - left_window
        e = i + right_window + 1
        start = max(s,0)
        end = min(e, array.__len__())
        for j in range(s, e):
            if j < start or j >= end:
                feature = np.concatenate((feature, np.zeros(array_set.__len__())), axis=0)
            else:
                try:
                    vector = tag_feat(array[j],array_set)
                    #print "vector: " + str(vector)
                except:
                    print "j: " + str(j)
                    print "start: " + str(start)
                    print "end: " + str(end)
                    print "s: " + str(s)
                    print "e: " + str(e)
                feature = np.concatenate((feature, vector), axis=0)
    else:
        start = max(i, 0)
        end = min(i + window + 1, array.__len__())
        for j in range(i, i + context + 1):
            if j < start or j >= end:
                feature = np.concatenate((feature, np.zeros(array_set.__len__())), axis=0)
            else:
                vector = tag_feat(array[j], array_set)
                feature = np.concatenate((feature, vector), axis=0)
    return feature


def MWE_candidates(lemma_arr, MWE_LEX, i, mwe_window, in_mwe, tag_distribution, tag_set):
    temp_features = []
    if in_mwe == 1 or tag_distribution == 1:
        candid_list = extractLexiconCandidates(lemma_arr, MWE_LEX)
        temp_dict = candid_list[0]
        mwe_lookup = ['B', 'I', 'b', 'i']
        try:
            left_mwe_window = mwe_window[0]
            right_mwe_window = mwe_window[1]
        except:
            left_mwe_window = mwe_window
            right_mwe_window = mwe_window
        start = max(i - left_mwe_window, 0)
        end = min(i + right_mwe_window + 1, lemma_arr.__len__())
        for j in range(i - left_mwe_window, i + right_mwe_window + 1):
            feature = []
            if j < start or j >= end:
                if in_mwe == 1:
                    feature = [-1]
                if tag_distribution == 1:
                    feature = np.zeros(tag_set.__len__())
            for key in temp_dict:
                temp_tag = tag_feat(temp_dict[key][i][1], tag_set)
                # print in_mwe_feat
                # print out_mwe_feat
                # in_mwe_feat = np.bitwise_or(in_mwe_feat,temp_tag)
                # out_mwe_feat = np.bitwise_and(out_mwe_feat, temp_tag)
                if mwe_lookup.__contains__(temp_dict[key][i][1]):
                    in_mwe_feat = temp_tag
                    break
                else:
                    in_mwe_feat = temp_tag
            # final_feat = [in_mwe_feat[0],in_mwe_feat[1],out_mwe_feat[2], in_mwe_feat[3], in_mwe_feat[4], out_mwe_feat[5]]
            final_feat = in_mwe_feat
            if in_mwe == 1:
                if final_feat[0] + final_feat[1] + final_feat[3] + final_feat[4] > 0:
                    feature = [1]
                else:
                    feature = [0]
                    # print "In test: " + str(in_mwe_feat)
            if tag_distribution == 1:
                feature = final_feat
            temp_features = np.concatenate((temp_features,feature), axis=0)
    return temp_features

def populate_features(word_dim, token_arr, lemma_arr, pos_arr, tag_arr, token_features, lemma_features,
                     pos_features, tag_features, context, window, feat_window, pos_window, tag_window, mwe_window, gen_vec, unk, unknown_token,
                     avg_tokens, avg_lemmas, normalize, in_mwe, tag_distribution, pos_tags, remove, model, sentence_embedding,
                     uni_skip, bi_skip, skipthoughts_tokens_vectors, skipthoughts_lemmas_vectors, sentence_counter,
                     con_win, k_win):
    feature_set = []
    for i in range(0, token_arr.__len__()):
        feature = []
        if token_features == 1:
            temp = word_feature(token_arr, model, window, feat_window, i, context, word_dim, gen_vec, unk, unknown_token)
            feature = np.concatenate((feature, temp), axis=0)
            #temp = sub_context(token_arr, model, k_win, i, word_dim, gen_vec, unk)
            #feature = np.concatenate((feature, temp), axis=0)
            #print "In tokens: " + str(feature.__len__())
            #print "temp: " + str(temp.__len__())
        if lemma_features == 1:
            temp = word_feature(lemma_arr, model, window, feat_window, i, context, word_dim, gen_vec, unk, unknown_token)
            feature = np.concatenate((feature, temp), axis=0)
            #temp = sub_context(lemma_arr, model, k_win, i, word_dim, gen_vec, unk)
            #feature = np.concatenate((feature, temp), axis=0)
            #print "In lemmas: " + str(feature.__len__())
            #print "temp: " + str(temp.__len__())
        if pos_features == 1:
            temp = window_feature(pos_arr, pos_tags[0], pos_window, i, context)
            feature = np.concatenate((feature, temp), axis=0)
            #print "In pos_features: " + str(feature.__len__())
            #print "temp: " + str(temp.__len__())
        if tag_features == 1:
            temp = window_feature(tag_arr, tag_set, tag_window, i, context)
            feature = np.concatenate((feature, temp), axis=0)
            #print "In tag_features: " + str(feature.__len__())
            #print "temp: " + str(temp.__len__())
        temp = context2vec(avg_tokens, remove, model, token_arr, word_dim, normalize, sentence_embedding, gen_vec, i, uni_skip,
                           bi_skip, skipthoughts_tokens_vectors, sentence_counter, con_win, unk, unknown_token)
        feature = np.concatenate((feature, temp), axis=0)
        #print "In token_context: " + str(feature.__len__())
        #print "temp: " + str(temp.__len__())
        temp = context2vec(avg_lemmas, remove, model, lemma_arr, word_dim, normalize, sentence_embedding, gen_vec, i, uni_skip,
                        bi_skip, skipthoughts_lemmas_vectors, sentence_counter, con_win, unk, unknown_token)
        feature = np.concatenate((feature, temp), axis=0)
        #print "In lemma_context: " + str(feature.__len__())
        #print "temp: " + str(temp.__len__())

        temp = MWE_candidates(lemma_arr, MWE_LEX, i, mwe_window, in_mwe, tag_distribution, tag_set)
        feature = np.concatenate((feature, temp), axis=0)
            #print "In mwe: " + str(feature.__len__())
            #print "temp: " + str(temp_features.__len__())
        feature_set.append(feature)
    return feature_set

def nn_features(path, window, feat_window, pos_window, tag_window, mwe_window, k_win, word_dim, tag_set, model, status, ngram,
                tag_features, token_features, lemma_features, pos_features, sub_token_lemma,
                context, con_win, avg_tokens, avg_lemmas, normalize, debug_features, in_mwe, tag_distribution,
                separate_lexicons,  pos_tags, emission_arr, poshmm, alpha, sentence_embedding, gen_vec, unk, unknown_token, remove,
                skipthoughts_tokens_path = None, skipthoughts_lemmas_path = None, uni_skip = 0, bi_skip = 0):

    training_features = []
    training_labels = []
    features = []
    skipthoughts_tokens_vectors = get_skipthoughts()
    skipthoughts_lemmas_vectors = get_skipthoughts()
    if status == 0:
        global general_vec
        general_vec = []
        if poshmm == 0:
            pos, emi = get_pos_hmm(path, ngram, alpha)
        else:
            pos, emi = get_pos_poshmm(path, ngram, alpha)
        pos_tags.append(pos)
        emission_arr.append(emi)
        print pos_tags[0]
    training_file = open(path)
    line = training_file.readline()
    training_tags = []
    context_features = []
    feature_size = 0
    token_arr = []
    lemma_arr = []
    pos_arr = []
    tag_arr = []
    i1 = 0
    i2 = 0
    i3 = 0
    token_sentence = []
    lemma_sentence = []
    sentence_counter = -1
    feature_count = 0
    label_count = 0
    while line != '':
        if line != '\n':
            k = line.split('\t')
            token = k[1]
            lemma = k[2]
            pos = k[3]
            tag = k[4]
            offset = k[5]
            supersense = k[6]
            id = k[7]
            token_arr.append(token)
            lemma_arr.append(lemma)
            pos_arr.append(pos)
            tag_arr.append(tag)
            training_labels.append(tag_feat(tag,tag_set))
            label_count += 1
            training_tags.append(tag)

        else:
            sentence_counter += 1
            #print sentence_counter
            features = populate_features(word_dim, token_arr, lemma_arr, pos_arr, tag_arr, token_features, lemma_features,
                     pos_features, tag_features, context, window, feat_window, pos_window, tag_window, mwe_window, gen_vec, unk, unknown_token,
                     avg_tokens, avg_lemmas, normalize, in_mwe, tag_distribution, pos_tags, remove, model, sentence_embedding,
                     uni_skip, bi_skip, skipthoughts_tokens_vectors, skipthoughts_lemmas_vectors, sentence_counter,
                     con_win, k_win)
            for feature in features:
                if feature_size == 0:
                    feature_size = feature.__len__()
                    print "There are " + str(feature_size) + " features in total"
                else:
                    if feature.__len__() != feature_size:
                        print "Standard feature is " + str(feature_size) + " but encoutered " + str(feature.__len__())
                        assert feature.__len__() == feature_size
                training_features.append(feature)
                feature_count += 1
                if feature_count % 10000 == 0:
                    print "Extracted " + str(feature_count) + " features"
            token_arr = []
            lemma_arr = []
            pos_arr = []
            tag_arr = []
        line = training_file.readline()
    print "There are " + str(training_features.__len__()) + " features"
    print "There are " + str(training_labels.__len__()) + " labels"
    print "There are " + str(sentence_counter) + " sentences in the file"
    print get_stats(training_tags)
    return training_features, training_labels
