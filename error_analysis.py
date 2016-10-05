import numpy as np
import operator
import string

def confusion_stats(gold, prediction, tag_set):
    matrix = np.zeros([tag_set.__len__(),tag_set.__len__()])
    for i in range(0, gold.__len__()):
        matrix[tag_set.index(prediction[i])][tag_set.index(gold[i])] += 1
    #print matrix
    return matrix

################################################################################################

def error_stats(arrays, top_k, labels, predictions):
    assert arrays.__len__() == top_k.__len__()
    assert labels.__len__() == predictions.__len__()
    results_dict = []
    shallow_dict = []
    for i in range(0, arrays.__len__()):
        results_dict.append({})
        shallow_dict.append({})
    for i in range(0,labels.__len__()):
        for ii in range(0, labels[i].__len__()):
            gold = labels[i][ii]
            pred = predictions[i][ii]
            pair = (gold,pred)
            for j in range(0, arrays.__len__()):
                key = arrays[j][i][ii]
                if results_dict[j].__contains__(key):
                    deep_dict = results_dict[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results_dict[j][key] = {pair: 1}
                if gold != pred:
                    if shallow_dict[j].__contains__(key):
                        shallow_dict[j][key] += 1
                    else:
                        shallow_dict[j][key] = 1
    for i in range(0,arrays.__len__()):
        intermediate_result = []
        sorted_dict = sorted(shallow_dict[i].items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() -1 -j][0]
                values = results_dict[i][key]
                intermediate_result.append((key,values))
        for element in intermediate_result:
            print element
        print ""

def stats_by_percentage(arrays, top_k, cutoff, labels, predictions):
    assert arrays.__len__() == top_k.__len__()
    assert labels.__len__() == predictions.__len__()
    assert top_k.__len__() == cutoff.__len__()
    correct = []
    wrong = []
    results = []
    for i in range(0, arrays.__len__()):
        correct.append({})
        wrong.append({})
        results.append({})
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[i].__len__()):
            gold = labels[i][ii]
            pred = predictions[i][ii]
            pair = (gold, pred)
            for j in range(0, arrays.__len__()):
                key = arrays[j][i][ii]
                if results[j].__contains__(key):
                    deep_dict = results[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results[j][key] = {pair: 1}
                if gold != pred:
                    if wrong[j].__contains__(key):
                        wrong[j][key] += 1
                    else:
                        wrong[j][key] = 1
                else:
                    if correct[j].__contains__(key):
                        correct[j][key] += 1
                    else:
                        correct[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        ratio_dict = {}
        for key in wrong[i]:
            wrong_count = wrong[i][key]
            if correct[i].__contains__(key):
                correct_count = correct[i][key]
            else:
                correct_count = 0
            if wrong_count + correct_count >= cutoff[i]:
                ratio_dict[key] = wrong_count*1.0/(correct_count + wrong_count)
        sorted_dict = sorted(ratio_dict.items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""
        
################################################################################################

def get_outcome(word,feature):
    if feature == 'cap':
        if string.ascii_uppercase.__contains__(word[0]):
            return 1
        return 0
    elif feature == 'anycap':
        for char in word:
            if string.ascii_uppercase.__contains__(char):
                return 1
        return 0
    elif feature == 'upper':
        for char in word:
            if not string.ascii_uppercase.__contains__(char):
                return 0
        return 1
    elif feature == 'nonchar':
        for char in word:
            if not string.ascii_letters.__contains__(char):
                return 1
        return 0
    elif feature == 'alpha':
        for char in word:
            if string.digits.__contains__(char):
                return 1
        return 0
    elif feature == 'num':
        for char in word:
            if not string.digits.__contains__(char):
                return 0
        return 1
    elif feature == 'http':
        if word.find('http') != -1:
            return 1
        else:
            return 0

################################################################################################

def word_stats_by_percentage(arrays, top_k, cutoff, features, labels, predictions):
    assert arrays.__len__() == top_k.__len__()
    assert labels.__len__() == predictions.__len__()
    assert top_k.__len__() == cutoff.__len__()
    correct = []
    wrong = []
    results = []
    for i in range(0, arrays.__len__()):
        correct.append({})
        wrong.append({})
        results.append({})
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[i].__len__()):
            gold = labels[i][ii]
            pred = predictions[i][ii]
            pair = (gold, pred)
            for k in features:
                for j in range(0, arrays.__len__()):
                    element = get_outcome(arrays[j][i][ii],k)
                    if element == 1:
                        key = (k,element)
                        if results[j].__contains__(key):
                            deep_dict = results[j][key]
                            if deep_dict.__contains__(pair):
                                deep_dict[pair] += 1
                            else:
                                deep_dict[pair] = 1
                        else:
                            results[j][key] = {pair: 1}
                        if gold != pred:
                            if wrong[j].__contains__(key):
                                wrong[j][key] += 1
                            else:
                                wrong[j][key] = 1
                        else:
                            if correct[j].__contains__(key):
                                correct[j][key] += 1
                            else:
                                correct[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        ratio_dict = {}
        for key in wrong[i]:
            wrong_count = wrong[i][key]
            if correct[i].__contains__(key):
                correct_count = correct[i][key]
            else:
                correct_count = 0
            if wrong_count + correct_count >= cutoff[i]:
                ratio_dict[key] = wrong_count*1.0/(correct_count + wrong_count)
        sorted_dict = sorted(ratio_dict.items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""


def word_stats(arrays, top_k, features, labels, predictions):
    assert arrays.__len__() == top_k.__len__()
    assert labels.__len__() == predictions.__len__()
    wrong = []
    results = []
    for i in range(0, arrays.__len__()):
        wrong.append({})
        results.append({})
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[i].__len__()):
            gold = labels[i][ii]
            pred = predictions[i][ii]
            if gold != pred:
                pair = (gold, pred)
                for k in features:
                    for j in range(0, arrays.__len__()):
                        element = get_outcome(arrays[j][i][ii],k)
                        if element == 1:
                            key = (k,element)
                            if results[j].__contains__(key):
                                deep_dict = results[j][key]
                                if deep_dict.__contains__(pair):
                                    deep_dict[pair] += 1
                                else:
                                    deep_dict[pair] = 1
                            else:
                                results[j][key] = {pair: 1}
                            if wrong[j].__contains__(key):
                                wrong[j][key] += 1
                            else:
                                wrong[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        sorted_dict = sorted(wrong[i].items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""

################################################################################################

def analyze_pair_by_percentage(array1, array2, top_k, cutoff, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert array1.__len__() == array2.__len__()
    correct = {}
    wrong = {}
    results = {}
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[i].__len__()):
            gold = labels[i][ii]
            pred = predictions[i][ii]
            pair = (gold, pred)
            target_pair = (array1[i][ii], array2[i][ii])
            if results.__contains__(target_pair):
                deep_dict = results[target_pair]
                if deep_dict.__contains__(pair):
                    deep_dict[pair] += 1
                else:
                    deep_dict[pair] = 1
            else:
                results[target_pair] = {pair: 1}
            if gold != pred:
                if wrong.__contains__(target_pair):
                    wrong[target_pair] += 1
                else:
                    wrong[target_pair] = 1
            else:
                if correct.__contains__(target_pair):
                    correct[target_pair] += 1
                else:
                    correct[target_pair] = 1
    intermediate_result = []
    ratio_dict = {}
    for key in wrong:
        wrong_count = wrong[key]
        if correct.__contains__(key):
            correct_count = correct[key]
        else:
            correct_count = 0
        if wrong_count + correct_count >= cutoff:
            ratio_dict[key] = wrong_count*1.0/(correct_count + wrong_count)
    sorted_dict = sorted(ratio_dict.items(), key=operator.itemgetter(1))
    for j in range(0, top_k):
        if sorted_dict.__len__() - 1 - j >= 0:
            key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
            values = results[key]
            intermediate_result.append((key, values))
    for element in intermediate_result:
        print element
    print ""

def analyze_pair(array1, array2, top_k, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert array1.__len__() == array2.__len__()
    wrong = {}
    results = {}
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[i].__len__()):
            gold = labels[i][ii]
            pred = predictions[i][ii]
            pair = (gold, pred)
            target_pair = (array1[i][ii], array2[i][ii])
            if results.__contains__(target_pair):
                deep_dict = results[target_pair]
                if deep_dict.__contains__(pair):
                    deep_dict[pair] += 1
                else:
                    deep_dict[pair] = 1
            else:
                results[target_pair] = {pair: 1}
            if gold != pred:
                if wrong.__contains__(target_pair):
                    wrong[target_pair] += 1
                else:
                    wrong[target_pair] = 1
    intermediate_result = []
    sorted_dict = sorted(wrong.items(), key=operator.itemgetter(1))
    for j in range(0, top_k):
        if sorted_dict.__len__() - 1 - j >= 0:
            key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
            values = results[key]
            intermediate_result.append((key, values))
    for element in intermediate_result:
        print element
    print ""

################################################################################################

def analyze_branches(array1, array2,  criteria, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert array1.__len__() == array2.__len__()
    result = {}
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[i].__len__()):
            if array1[i][ii] == criteria:
                gold = labels[i][ii]
                pred = predictions[i][ii]
                pair = (gold, pred)
                pair2 = (array1[i][ii], array2[i][ii])
                if result.__contains__(pair2):
                    deep_dict = result[pair2]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    result[pair2] = {pair: 1}
    for element in result:
        print str(element) + ": " + str(result[element])
    print ""

################################################################################################

def analyze_window_by_precentage(arrays, top_k, cutoff, window, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert arrays.__len__() == cutoff.__len__()
    assert top_k.__len__() == cutoff.__len__()
    results = []
    wrong = []
    correct = []
    for i in range(0, arrays.__len__()):
        results.append({})
        wrong.append({})
        correct.append({})
    start = 1
    end = window + 1
    if window < 0:
        start = window
        end = 0
    for i in range(0,labels.__len__()):
        for ii in range(0, labels[0].__len__()):
            pred = predictions[i][ii]
            gold = labels[i][ii]
            pair = (gold, pred)
            for j in range(0, arrays.__len__()):
                key = ()
                for k in range(start, end):
                    if ii + k < 0 or ii + k >= arrays[j][i].__len__():
                        key += ('NULL',)
                    else:
                        key += (arrays[j][i][ii + k],)
                if results[j].__contains__(key):
                    deep_dict = results[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results[j][key] = {pair: 1}
                if gold != pred:
                    if wrong[j].__contains__(key):
                        wrong[j][key] += 1
                    else:
                        wrong[j][key] = 1
                else:
                    if correct[j].__contains__(key):
                        correct[j][key] += 1
                    else:
                        correct[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        ratio_dict = {}
        for key in wrong[i]:
            wrong_count = wrong[i][key]
            if correct[i].__contains__(key):
                correct_count = correct[i][key]
            else:
                correct_count = 0
            if wrong_count + correct_count >= cutoff[i]:
                ratio_dict[key] = wrong_count * 1.0 / (correct_count + wrong_count)
        sorted_dict = sorted(ratio_dict.items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""


def analyze_window(arrays, top_k, cutoff, window, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert arrays.__len__() == cutoff.__len__()
    assert top_k.__len__() == cutoff.__len__()
    results = []
    wrong = []
    for i in range(0, arrays.__len__()):
        results.append({})
        wrong.append({})
    start = 1
    end = window + 1
    if window < 0:
        start = window
        end = 0
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[0].__len__()):
            pred = predictions[i][ii]
            gold = labels[i][ii]
            pair = (gold, pred)
            for j in range(0, arrays.__len__()):
                key = ()
                for k in range(start, end):
                    if ii + k < 0 or ii + k >= arrays[j][i].__len__():
                        key += ('NULL',)
                    else:
                        key += (arrays[j][i][ii + k],)
                if results[j].__contains__(key):
                    deep_dict = results[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results[j][key] = {pair: 1}
                if gold != pred:
                    if wrong[j].__contains__(key):
                        wrong[j][key] += 1
                    else:
                        wrong[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        sorted_dict = sorted(wrong[i].items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""

###############################################################################################3

def analyze_ngrams_by_precentage(arrays, top_k, cutoff, window, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert arrays.__len__() == cutoff.__len__()
    assert top_k.__len__() == cutoff.__len__()
    results = []
    wrong = []
    correct = []
    for i in range(0, arrays.__len__()):
        results.append({})
        wrong.append({})
        correct.append({})
    start = 0
    end = window + 1
    if window < 0:
        start = window
        end = 1
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[0].__len__()):
            pred = predictions[i][ii]
            gold = labels[i][ii]
            pair = (gold, pred)
            for j in range(0, arrays.__len__()):
                key = ()
                for k in range(start, end):
                    if ii + k < 0 or ii + k >= arrays[j][i].__len__():
                        key += ('NULL',)
                    else:
                        key += (arrays[j][i][ii + k],)
                if results[j].__contains__(key):
                    deep_dict = results[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results[j][key] = {pair: 1}
                if gold != pred:
                    if wrong[j].__contains__(key):
                        wrong[j][key] += 1
                    else:
                        wrong[j][key] = 1
                else:
                    if correct[j].__contains__(key):
                        correct[j][key] += 1
                    else:
                        correct[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        ratio_dict = {}
        for key in wrong[i]:
            wrong_count = wrong[i][key]
            if correct[i].__contains__(key):
                correct_count = correct[i][key]
            else:
                correct_count = 0
            if wrong_count + correct_count >= cutoff[i]:
                ratio_dict[key] = wrong_count * 1.0 / (correct_count + wrong_count)
        sorted_dict = sorted(ratio_dict.items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""

def analyze_ngrams(arrays, top_k, cutoff, window, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert arrays.__len__() == cutoff.__len__()
    assert top_k.__len__() == cutoff.__len__()
    results = []
    wrong = []
    for i in range(0, arrays.__len__()):
        results.append({})
        wrong.append({})
    start = 0
    end = window
    if window < 0:
        start = window
        end = 1
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[0].__len__()):
            pred = predictions[i][ii]
            gold = labels[i][ii]
            pair = (gold, pred)
            for j in range(0, arrays.__len__()):
                key = ()
                for k in range(start, end):
                    if ii + k < 0 or ii + k >= arrays[j][i].__len__():
                        key += ('NULL',)
                    else:
                        key += (arrays[j][i][ii + k],)
                if results[j].__contains__(key):
                    deep_dict = results[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results[j][key] = {pair: 1}
                if gold != pred:
                    if wrong[j].__contains__(key):
                        wrong[j][key] += 1
                    else:
                        wrong[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        sorted_dict = sorted(wrong[i].items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""

################################################################################################

def analyze_characters_by_precentage(arrays, top_k, cutoff, window, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert arrays.__len__() == cutoff.__len__()
    assert top_k.__len__() == cutoff.__len__()
    results = []
    wrong = []
    correct = []
    for i in range(0, arrays.__len__()):
        results.append({})
        wrong.append({})
        correct.append({})
    start = 0
    end = window
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[0].__len__()):
            pred = predictions[i][ii]
            gold = labels[i][ii]
            pair = (gold, pred)
            for j in range(0, arrays[i][ii].__len__()):
                key = ()
                string_length = arrays[j][i][ii].__len__()
                if window < 0:
                    start = string_length + window
                    end = string_length
                for k in range(start, end):
                    if k < 0 or k >= string_length:
                        key += ('NULL',)
                    else:
                        key += (arrays[j][i][ii][k],)
                if results[j].__contains__(key):
                    deep_dict = results[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results[j][key] = {pair: 1}
                if gold != pred:
                    if wrong[j].__contains__(key):
                        wrong[j][key] += 1
                    else:
                        wrong[j][key] = 1
                else:
                    if correct[j].__contains__(key):
                        correct[j][key] += 1
                    else:
                        correct[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        ratio_dict = {}
        for key in wrong[i]:
            wrong_count = wrong[i][key]
            if correct[i].__contains__(key):
                correct_count = correct[i][key]
            else:
                correct_count = 0
            if wrong_count + correct_count >= cutoff[i]:
                ratio_dict[key] = wrong_count * 1.0 / (correct_count + wrong_count)
        sorted_dict = sorted(ratio_dict.items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""

def analyze_characters(arrays, top_k, cutoff, window, labels, predictions):
    assert labels.__len__() == predictions.__len__()
    assert arrays.__len__() == cutoff.__len__()
    assert top_k.__len__() == cutoff.__len__()
    results = []
    wrong = []
    for i in range(0, arrays.__len__()):
        results.append({})
        wrong.append({})
    start = 0
    end = window
    for i in range(0, labels.__len__()):
        for ii in range(0, labels[0].__len__()):
            pred = predictions[i][ii]
            gold = labels[i][ii]
            pair = (gold, pred)
            for j in range(0, arrays.__len__()):
                key = ()
                string_length = arrays[j][i][ii].__len__()
                if window < 0:
                    start = string_length + window
                    end = string_length
                for k in range(start, end):
                    if k < 0 or k >= string_length:
                        key += ('NULL',)
                    else:
                        key += (arrays[j][i][ii][k],)
                if results[j].__contains__(key):
                    deep_dict = results[j][key]
                    if deep_dict.__contains__(pair):
                        deep_dict[pair] += 1
                    else:
                        deep_dict[pair] = 1
                else:
                    results[j][key] = {pair: 1}
                if gold != pred:
                    if wrong[j].__contains__(key):
                        wrong[j][key] += 1
                    else:
                        wrong[j][key] = 1
    for i in range(0, arrays.__len__()):
        intermediate_result = []
        sorted_dict = sorted(wrong[i].items(), key=operator.itemgetter(1))
        for j in range(0, top_k[i]):
            if sorted_dict.__len__() - 1 - j >= 0:
                key = sorted_dict[sorted_dict.__len__() - 1 - j][0]
                values = results[i][key]
                intermediate_result.append((key, values))
        for element in intermediate_result:
            print element
        print ""
