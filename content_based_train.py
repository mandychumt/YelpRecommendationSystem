from pyspark import SparkContext
import json
import re
import math
import time
import sys

def generate_stopwords(stopwords_file):
    fh = open(stopwords_file)
    stopwords = [word.strip('\n') for word in fh]
    stopwords = set(stopwords)
    stopwords.add('ve')
    fh.close()
    return stopwords

def remove_elements_and_split(business_id, texts, stopwords):
    words = []
    for text in texts:
        text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~0-9]|\n', ' ', text)
        for word in text.lower().split(' '):
            if len(word) > 1 and word not in stopwords:
                words.append(word)
    return (business_id, words)

def generate_word_count(business_id, words, with_id):
    result = []
    for word in words:
        if with_id:
            result.append((word, (1, business_id)))
        else:
            result.append((word, 1))
    return result

def get_term_freq(business_id, doc, rare_words):
    word_freq = {}
    max_freq = 0
    for word in doc:
        if word not in rare_words:
            word_freq[word] = word_freq.get(word, 0) + 1
            max_freq = max(word_freq[word], max_freq)
    term_freq = []
    for word, freq in word_freq.items():
        term_freq.append([word, (1, [(freq / max_freq, business_id)])])
    return term_freq


def main():
	start = time.time()

	train_file = sys.argv[1]
	model_file = sys.argv[2]
	stopwords_file = sys.argv[3]

	sc = SparkContext()
	sc.setLogLevel("ERROR")

	train_reivew = sc.textFile(train_file) \
                    .map(lambda line: json.loads(line))

	business_texts = train_reivew \
	                .map(lambda line: (line['business_id'], line['text'])) \
	                .groupByKey()
	doc_count = business_texts.count()

	# business_review_words: (business_id, [words after removing punctuations, numbers, stopwords from all its reviews])
	stopwords = generate_stopwords(stopwords_file)
	business_review_words = business_texts \
	                    .map(lambda line: remove_elements_and_split(line[0], line[1], stopwords))

	# remove rare_words (count <= 0.0001% of the totalwords)
	all_words = business_review_words \
	            .flatMap(lambda line: generate_word_count(line[0], line[1], False)) \
	            .reduceByKey(lambda x, y: x + y)
	words_count = all_words.values().sum()
	threshold = words_count * 1e-6
	rare_words = all_words \
	            .filter(lambda line: line[1] < threshold) \
	            .map(lambda line: line[0]) \
	            .collect()
	all_words = None
	rare_words = set(rare_words)

	# calculate TF-IDF and create business profile
	business_profile = business_review_words \
	                .flatMap(lambda line: get_term_freq(line[0], line[1], rare_words)) \
	                .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
	                .flatMap(lambda line: [
	                    [business_id, (math.log(doc_count / line[1][0]) * term_freq, line[0])] for term_freq, business_id in line[1][1]
	                    ]) \
	                .groupByKey() \
	                .mapValues(lambda line: [t[1] for t in sorted(line)[:200]])
	rare_words.clear()

	# create user prifile
	user_profile = train_reivew \
	            .map(lambda line: (line['business_id'], line['user_id'])) \
	            .join(business_profile) \
	            .map(lambda line: (line[1][0], line[1][1])) \
	            .reduceByKey(lambda x, y: x + y) \
	            .mapValues(lambda line: set(line))

	# output model file
	business_profile_ls = business_profile.collect()
	fh = open(model_file, 'w')
	for business_id, words in business_profile_ls:
	    content = {'business_id': business_id, 'profile': words}
	    json.dump(content, fh)
	    fh.write('\n')
	business_profile_ls.clear()

	user_profile_ls = user_profile.collect()
	for user_id, words in user_profile_ls:
	    content = {'user_id': user_id, 'profile': list(words)}
	    json.dump(content, fh)
	    fh.write('\n')

	fh.close()

	print("Duration: %s" % (time.time() - start))


if __name__ == "__main__":
    main()