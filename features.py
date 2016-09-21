
import numpy as np
import extract_mwes
import skipthoughts

MWE_LEX = "/home/waseem/Downloads/pysupersensetagger-2.0/mwelex/"

start_tag = 'start'
end_tag = 'end'
tag_set = ['B', 'I', 'O', 'b', 'i', 'o']

def get_pos_hmm(path, ngram, binary_transitions, alpha):
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
                    if binary_transitions == 1:
                        temp[tag_tuple] = 1
                    else:
                        temp[tag_tuple] = (emission_count[tag_tuple] + alpha)*1.0/(emission_array[i-1][tag_tuple[:i]] + alpha*emission_count.__len__())
                else:
                    temp[tag_tuple[:i+1]] = emission_count[tag_tuple]
            else:
                if i > 0:
                    if binary_transitions == 1:
                        temp[tag_tuple] = 1
                    else:
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

def get_vector(model,word_arr, index, con_win, word_dim, normalize, one_way, gen_vec, remove_word):
    resultant = 0
    if one_way == 0:
        min_index = index - con_win
        min_index = max(min_index,0)
        max_index = index + con_win + 1
        max_index = min(max_index, word_arr.__len__())
        temp = np.zeros(word_dim)
        norm = 0
        for i in range(min_index, max_index):
            if i != remove_word:
                if model.__contains__(word_arr[i]):
                    temp += model[word_arr[i]]
                    norm += 1
                elif model.__contains__(word_arr[i].lower()):
                    temp += model[word_arr[i].lower()]
                    norm += 1
                elif gen_vec == 1:
                    temp += general_vec
        if normalize == 1 and np.sum(temp) > 0 and gen_vec == 0:
            temp /= norm
        else:
            temp /= con_win
        resultant = temp
    if one_way == 1:
        temp = np.zeros(word_dim)
        norm = 0
        if con_win + index > word_arr.__len__():
            con_win = word_arr.__len__() - index
        for i in range(index, index + con_win):
            if i != remove_word:
                if model.__contains__(word_arr[i]):
                    temp += model[word_arr[i]]
                    norm += 1
                elif model.__contains__(word_arr[i].lower()):
                    temp += model[word_arr[i].lower()]
                    norm += 1
                elif gen_vec == 1:
                    temp += general_vec
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

def get_skipthoughts(path):
    skipthoughts_vectors = []
    if path != None:
        skipthoughts_file = open(path)
        skipthoughts_line = skipthoughts_file.readline()
        while skipthoughts_line != '':
            temp_vec = read_vector(skipthoughts_line)
            skipthoughts_vectors.append(temp_vec)
            skipthoughts_line = skipthoughts_file.readline()
    return skipthoughts_vectors

def append_features(model, word, word_dim, gen_vec, unk):
    feature = []
    unk_array = []
    vector = np.zeros(word_dim)
    if model.__contains__(word):
        vector = model[word]
    if model.__contains__(word.lower()):
        vector = model[word.lower()]
        if unk == 1:
            unk_array.append(0)
    else:
        # print "feature size was " + str(features.__len__())
        if gen_vec == 1:
            vector = general_vec
        else:
            vector = np.zeros(word_dim)
        if unk == 1:
            unk_array.append(1)
        # print "feature size is " + str(features.__len__())
    feature = np.concatenate((feature, vector), axis=0)
    #print "feature length: " + str(features.__len__())
    feature = np.concatenate((feature, unk_array), axis=0)
    #print "feature length2: " + str(features.__len__())
    return feature


def context2vec(avg_tokens, remove, model, word_arr, word_dim, normalize, sentence_embedding, gen_vec, index, uni_skip, bi_skip, skipthoughts_word_vectors, sentence_counter, con_win):
    word_vec = []
    if avg_tokens == 1:
        if remove == 1:
            word_vec = get_vector(model, word_arr, index, con_win, word_dim, normalize, sentence_embedding,
                                   gen_vec, index)
        else:
            # print sentence_counter
            if uni_skip == 1:
                word_vec += skipthoughts_word_vectors[sentence_counter][0:2400]
            if bi_skip == 1:
                word_vec += skipthoughts_word_vectors[sentence_counter][2401:4800]
            if uni_skip == 0 and bi_skip == 0:
                word_vec = get_vector(model, word_arr, index, con_win, word_dim, normalize, sentence_embedding,
                                       gen_vec, -1)
    return word_vec


def word_feature(word_arr, model, window, i, context, word_dim, gen_vec, unk):
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
                feature = np.concatenate((feature, np.zeros(word_dim)), axis=0)
                #print "vector: " + str(np.zeros(word_dim))
            else:
                vector = append_features(model, word_arr[j], word_dim, gen_vec, unk)
                #print "vector: " + str(vector)
                feature = np.concatenate((feature, vector), axis=0)
    else:
        start = max(i, 0)
        end = min(i + window + 1, word_arr.__len__())
        for j in range(i, i + context + 1):
            if j < start or j >= end:
                feature = np.concatenate((feature, np.zeros(word_dim)), axis=0)
            else:
                vector = append_features(model, word_arr[j], word_dim, gen_vec, unk)
                feature = np.concatenate((feature, vector), axis=0)
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
                     pos_features, tag_features, context, window, pos_window, tag_window, mwe_window, gen_vec, unk, avg_tokens,
                     avg_lemmas, normalize, in_mwe, tag_distribution, pos_tags, remove, model, sentence_embedding,
                     uni_skip, bi_skip, skipthoughts_tokens_vectors, skipthoughts_lemmas_vectors, sentence_counter,
                     con_win):
    feature_set = []
    for i in range(0, token_arr.__len__()):
        feature = []
        if token_features == 1:
            temp = word_feature(token_arr, model, window, i, context, word_dim, gen_vec, unk)
            feature = np.concatenate((feature, temp), axis=0)
            #print "In tokens: " + str(feature.__len__())
            #print "temp: " + str(temp.__len__())
        if lemma_features == 1:
            temp = word_feature(lemma_arr, model, window, i, context, word_dim, gen_vec, unk)
            feature = np.concatenate((feature, temp), axis=0)
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
                           bi_skip, skipthoughts_tokens_vectors, sentence_counter, con_win)
        feature = np.concatenate((feature, temp), axis=0)
        #print "In token_context: " + str(feature.__len__())
        #print "temp: " + str(temp.__len__())
        temp = context2vec(avg_lemmas, remove, model, lemma_arr, word_dim, normalize, sentence_embedding, gen_vec, i, uni_skip,
                        bi_skip, skipthoughts_lemmas_vectors, sentence_counter, con_win)
        feature = np.concatenate((feature, temp), axis=0)
        #print "In lemma_context: " + str(feature.__len__())
        #print "temp: " + str(temp.__len__())

        temp = MWE_candidates(lemma_arr, MWE_LEX, i, mwe_window, in_mwe, tag_distribution, tag_set)
        feature = np.concatenate((feature, temp), axis=0)
            #print "In mwe: " + str(feature.__len__())
            #print "temp: " + str(temp_features.__len__())
        feature_set.append(feature)
    return feature_set

def nn_features(path, window, pos_window, tag_window, mwe_window, word_dim, tag_set, model, status, ngram, binary_transitions, tag_features, token_features,
                lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize,
                debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, alpha,
                sentence_embedding, gen_vec, unk, remove, skipthoughts_tokens_path = None, skipthoughts_lemmas_path = None,
                uni_skip = 0, bi_skip = 0):

    training_features = []
    training_labels = []
    features = []
    skipthoughts_tokens_vectors = get_skipthoughts(skipthoughts_tokens_path)
    skipthoughts_lemmas_vectors = get_skipthoughts(skipthoughts_lemmas_path)
    if status == 0:
        global general_vec
        general_vec = []
        pos, emi = get_pos_hmm(path, ngram, binary_transitions, alpha)
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
                     pos_features, tag_features, context, window, pos_window, tag_window, mwe_window, gen_vec, unk, avg_tokens,
                     avg_lemmas, normalize, in_mwe, tag_distribution, pos_tags, remove, model, sentence_embedding,
                     uni_skip, bi_skip, skipthoughts_tokens_vectors, skipthoughts_lemmas_vectors, sentence_counter,
                     con_win)
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


def rnn_features(path, window, word_dim, tag_set, model, status, ngram, binary_transitions, tag_features, token_features,
                lemma_features, pos_features, sub_token_lemma, context, con_win, avg_tokens, avg_lemmas, normalize,
                debug_features, in_mwe, tag_distribution, separate_lexicons,  pos_tags, emission_arr, alpha,
                sentence_embedding, gen_vec, unk, remove, skipthoughts_tokens_path = None, skipthoughts_lemmas_path = None,
                uni_skip = 0, bi_skip = 0):

    training_features = []
    training_labels = []
    features = []
    skipthoughts_tokens_vectors = get_skipthoughts(skipthoughts_tokens_path)
    skipthoughts_lemmas_vectors = get_skipthoughts(skipthoughts_lemmas_path)
    if status == 0:
        global general_vec
        general_vec = []
        pos, emi = get_pos_hmm(path, ngram, binary_transitions, alpha)
        pos_tags.append(pos)
        emission_arr.append(emi)
        print pos_tags[0]
    training_file = open(path)
    line = training_file.readline()
    factor = 0
    if token_features == 1:
        factor += word_dim
        if unk == 1:
            factor += 1
    if tag_features == 1:
        factor += tag_set.__len__()
    if lemma_features == 1:
        factor += word_dim
        if unk == 1:
            factor += 1
    if pos_features == 1:
        factor += pos_tags[0].__len__()
    for win in range(0,window):
        features = np.concatenate((features,np.zeros(word_dim)),axis=0)
        if lemma_features == 1:
            features = np.concatenate((features,np.zeros(word_dim)),axis=0)
        if pos_features == 1:
            features = np.concatenate((features,np.zeros(pos_tags[0].__len__())),axis=0)
        if tag_features == 1:
            temp = tag_feat('R',tag_set)
            features = np.concatenate((features,temp),axis=0)
    training_tags = []
    context_features = []
    bias = 0
    token_arr = []
    lemma_arr = []
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
            if gen_vec == 1 and general_vec.__len__() == 0:
                general_vec = get_general_vec(model, word_dim)
            #features = append_features(features, model, [token, lemma], sub_token_lemma, word_dim, gen_vec, unk, np.subtract)
            features = append_features(features, model, [token], token_features, word_dim, gen_vec, unk, np.add)
            features = append_features(features, model, [lemma], lemma_features, word_dim, gen_vec, unk, np.add)
            #print "general_vec: " + str(general_vec)
            if pos_features == 1:
                pos_arr = np.zeros(pos_tags[0].__len__())
                pos_arr[pos_tags[0].index(pos)] = 1
                features = np.concatenate((features,pos_arr),axis=0)
            if tag_features == 1:
                features = np.concatenate((features,tag_feat('R',tag_set)),axis=0)
            training_features.append(features)
            training_labels.append(tag_feat(tag,tag_set))
            label_count += 1
            training_tags.append(tag)
            features = features[factor:]
            if tag_features == 1:
                if status == 1:
                    #features = np.concatenate((features,tag_feat('R',tag_set)),axis=0)
                    features[features.__len__() - tag_set.__len__():] = tag_feat('R',tag_set)
                else:
                    #features = np.concatenate((features,tag_feat(tag,tag_set)),axis=0)
                    features[features.__len__() - tag_set.__len__():] = tag_feat(tag,tag_set)
            # if model.__contains__(token):
            #     features = np.concatenate((features[factor:], model[token]),axis=0)
            # else:
            #     features = np.concatenate((features[factor:], np.zeros(word_dim)),axis=0)
        else:
            sentence_counter += 1
            begin = bias
            if context == 1:
                begin += window/2
                current_index = -1
                if begin < training_features.__len__():
                    for i in range(begin,training_features.__len__()):
                        feature_count += 1
                        if feature_count % 10000 == 0:
                            print "Extracted " + str(feature_count) + " features in total"
                        temp_features = training_features[i]
                        if con_win > 0:
                            temp_features = context2vec(temp_features, avg_tokens, remove, model, token_arr, word_dim,
                                            normalize, sentence_embedding, gen_vec, i - begin, uni_skip, bi_skip,
                                            skipthoughts_tokens_vectors, sentence_counter, con_win)
                            temp_features = context2vec(temp_features, avg_lemmas, remove, model, lemma_arr, word_dim,
                                            normalize, sentence_embedding, gen_vec, i - begin, uni_skip, bi_skip,
                                            skipthoughts_lemmas_vectors, sentence_counter, con_win)
                        current_index = i - begin
                        context_features.append(temp_features)
                        i1 += 1
                else:
                    for i in range(training_features.__len__(),begin):
                        feature_count += 1
                        features = np.concatenate((features,np.zeros(factor)),axis=0)
                        #display_features(features)
                        features = features[factor:]
                        #print "features length special = " + str(features.__len__())
                        i2 += 1
                current_index += 1
                if training_features.__len__() - begin < 0:
                    ulimit = training_features.__len__() - bias
                    #print "ulimit: " + str(ulimit)
                else:
                    ulimit = window/2
                for i in range(0,ulimit):
                    features = np.concatenate((features,np.zeros(factor)),axis=0)
                    feature_count += 1
                    if con_win > 0:
                        features = context2vec(features, avg_tokens, remove, model, token_arr, word_dim,
                                               normalize, sentence_embedding, gen_vec, current_index, uni_skip,
                                               bi_skip, skipthoughts_tokens_vectors, sentence_counter, con_win)
                        features = context2vec(features, avg_lemmas, remove, model, lemma_arr, word_dim,
                                                normalize, sentence_embedding, gen_vec, current_index, uni_skip,
                                               bi_skip, skipthoughts_lemmas_vectors, sentence_counter, con_win)
                            #print "skip-thought vectors created"
                    context_features.append(features)
                    temp_factor = features.__len__() - ((avg_tokens + avg_lemmas)*word_dim)
                    features = features[factor:temp_factor]
                    current_index += 1
                    i3 += 1
                if debug_features == 1:
                    if current_index != token_arr.__len__():
                        print "The current index: " + str(current_index)
                        print "Tokens: " + str(token_arr)
                    #print "features length = " + str(features.__len__())
                #print "There are " + str(context_features[i].__len__()) + " context features"
            #if features.__len__() != feature_vector_size:
            #    print "The feature sizes are not equal"
            #    display_features(features)
            #    print str(features.__len__()) + " instead of " + str(feature_vector_size)
            #print "There you go" + 5
            else:
                for i in range(begin,training_features.__len__()):
                    feature_count += 1
                    context_features.append(training_features[i])
            if in_mwe == 1 or tag_distribution == 1:
                candid_list = extractLexiconCandidates(lemma_arr,MWE_LEX)
                temp_dict = candid_list[0]
            for i in range(bias,context_features.__len__()):
                temp_features = context_features[i]
                if context == 1:
                    start = int(window)/-2
                    end = int(window)/2
                else:
                    start = -1*int(window)
                    end = 0
                for token_index in range(start,end+1):
                    base = i - bias
                    if base + token_index < 0 or base + token_index >= token_arr.__len__():
                        #temp_features = np.concatenate((temp_features,np.zeros(tag_set.__len__())), axis=0)
                        temp_features = np.concatenate((temp_features,np.zeros(1)), axis=0)
                    else:
                        if separate_lexicons == 0:
                            mwe_lookup = ['B','I','b','i']
                            for key in temp_dict:
                                temp_tag = tag_feat(temp_dict[key][base + token_index][1], tag_set)
                                #print in_mwe_feat
                                #print out_mwe_feat
                                #in_mwe_feat = np.bitwise_or(in_mwe_feat,temp_tag)
                                #out_mwe_feat = np.bitwise_and(out_mwe_feat, temp_tag)
                                if mwe_lookup.__contains__(temp_dict[key][base + token_index][1]):
                                    in_mwe_feat = temp_tag
                                    break
                                elif temp_dict[key][base + token_index][1] == 'o':
                                    in_mwe_feat = temp_tag
                                else:
                                    in_mwe_feat = temp_tag
                            #final_feat = [in_mwe_feat[0],in_mwe_feat[1],out_mwe_feat[2], in_mwe_feat[3], in_mwe_feat[4], out_mwe_feat[5]]
                            final_feat = in_mwe_feat
                            if in_mwe == 1:
                                if final_feat[0] + final_feat[1] + final_feat[3] + final_feat[4] > 0:
                                    temp_features = np.concatenate((temp_features,[1]), axis=0)
                                else:
                                    temp_features = np.concatenate((temp_features,[0]), axis=0)
                                #print "In test: " + str(in_mwe_feat)
                            if tag_distribution == 1:
                                temp_features = np.concatenate((temp_features,final_feat), axis=0)
                                #temp_features = np.concatenate((temp_features,out_mwe_feat), axis=0)
                        else:
                            in_mwe_feat = [1,1,1,1,1,1]
                            out_mwe_feat = [0,0,0,0,0,0]
                            mwe_lookup = ['B','I','b','i']
                            for j in range(0,temp_dict.__len__()):
                                key = temp_dict.keys()[j]
                                if mwe_lookup.__contains__(temp_dict[key][i - begin][1]):
                                    in_mwe_feat[j] = 1
                                else:
                                    out_mwe_feat[j] = 1
                            if in_mwe == 1:
                                temp_features = np.concatenate((temp_features,in_mwe_feat), axis=0)
                            if tag_distribution == 1:
                                temp_features = np.concatenate((temp_features,in_mwe_feat), axis=0)
                                temp_features = np.concatenate((temp_features,out_mwe_feat), axis=0)
                context_features[i] = temp_features
            #print "Lexicon Candid output: " + str(kkk)
            #print "h = " + 5
            features = []
            token_arr = []
            lemma_arr = []
            bias = training_features.__len__()
            for win in range(0,window):
                features = np.concatenate((features,np.zeros(word_dim)),axis=0)
                if lemma_features == 1:
                    features = np.concatenate((features,np.zeros(word_dim)),axis=0)
                if pos_features == 1:
                    features = np.concatenate((features,np.zeros(pos_tags[0].__len__())),axis=0)
                if tag_features == 1:
                    features = np.concatenate((features,tag_feat('R',tag_set)),axis=0)
        line = training_file.readline()
    print get_stats(training_tags)
    #print "i1: " + str(i1) + ", i2: " + str(i2) + ", i3: " + str(i3)
    #print "Sum = " + str(i1 + i2 + i3)
    #if context == 1:
    print "There are " + str(context_features.__len__()) + " context features"
    print "There are " + str(training_labels.__len__()) + " context labels"
    return context_features, training_labels
    #return training_features, training_labels
