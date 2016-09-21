import numpy as np
import operator
import string

def confusion_stats(gold, prediction, tag_set):
    matrix = np.zeros([tag_set.__len__(),tag_set.__len__()])
    for i in range(0, gold.__len__()):
        matrix[tag_set.index(prediction[i])][tag_set.index(gold[i])] += 1
    #print matrix
    return matrix

def error_stats(arrays, top_k, labels, predictions, tag_set):
    assert arrays.__len__() == top_k.__len__()
    assert labels.__len__() == predictions.__len__()
    results_dict = []
    shallow_dict = []
    for i in range(0, arrays.__len__()):
        results_dict.append({})
        shallow_dict.append({})
    for i in range(0,labels.__len__()):
        gold = tag_set[labels[i].index(1)]
        pred = predictions[i]
        pair = (gold,pred)
        for j in range(0, arrays.__len__()):
            if results_dict[j].__contains__(arrays[j][i]):
                deep_dict = results_dict[j][arrays[j][i]]
                if deep_dict.__contains__(pair):
                    deep_dict[pair] += 1
                else:
                    deep_dict[pair] = 1
            else:
                results_dict[j][arrays[j][i]] = {pair: 1}
            if gold != pred:
                if shallow_dict[j].__contains__(arrays[j][i]):
                    shallow_dict[j][arrays[j][i]] += 1
                else:
                    shallow_dict[j][arrays[j][i]] = 1
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

def stats_by_percentage(arrays, top_k, cutoff, labels, predictions, tag_set):
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
        gold = tag_set[labels[i].index(1)]
        pred = predictions[i]
        pair = (gold, pred)
        for j in range(0, arrays.__len__()):
            if results[j].__contains__(arrays[j][i]):
                deep_dict = results[j][arrays[j][i]]
                if deep_dict.__contains__(pair):
                    deep_dict[pair] += 1
                else:
                    deep_dict[pair] = 1
            else:
                results[j][arrays[j][i]] = {pair: 1}
            if gold != pred:
                if wrong[j].__contains__(arrays[j][i]):
                    wrong[j][arrays[j][i]] += 1
                else:
                    wrong[j][arrays[j][i]] = 1
            else:
                if correct[j].__contains__(arrays[j][i]):
                    correct[j][arrays[j][i]] += 1
                else:
                    correct[j][arrays[j][i]] = 1
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


def word_stats_by_percentage(arrays, top_k, cutoff, features, labels, predictions, tag_set):
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
        gold = tag_set[labels[i].index(1)]
        pred = predictions[i]
        pair = (gold, pred)
        for k in features:
            for j in range(0, arrays.__len__()):
                element = get_outcome(arrays[j][i],k)
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
