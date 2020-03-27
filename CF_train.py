from pyspark import SparkContext
import json
import math
import statistics
import random
import itertools
import time
import sys

def remove_duplicates(value):
    length = len(value)
    if len(set([t[0] for t in value])) == length:
        return value
    value.sort()
    result = []
    l = 0
    r = 1
    while r <= length:
        if r == length or value[l][0] != value[r][0]:
            result.append((value[l][0], statistics.mean([t[1] for t in value[l : r]])))
            l = r
        r += 1
    return result

def hash_func(param1, param2, row_count, v):
    return ((param1 * v + param2) % 10259) % row_count

def get_params(hash_num):
    params = []
    for i in range(hash_num):
        param = random.randint(0, 4000)
        while param in params:
            param = random.randint(0, 4000)
        params.append(param)
    return params

def generate_sigs(value, hash_num, row_count, params1, params2):
	sigs = [min([hash_func(params1[i], params2[i], row_count, row_index) for row_index in value]) for i in range(hash_num)]
	return tuple(sigs)

def get_bands(user_id, signatures, b, r):
    return [((i, ','.join(map(str, signatures[i * r: (i + 1) * r]))), user_id) for i in range(b)]

def get_candidate_pair(business_ids):
    if not business_ids or len(business_ids) < 2:
        return []
    pairs = []
    for pair in itertools.combinations(business_ids, 2):
        pairs.append(tuple(sorted(pair)))
    return pairs

def get_jaccard_sim(pair):
    set1 = set(pair[0])
    set2 = set(pair[1])
    size_intersection = len(set1 & set2)
    return size_intersection / (len(set1) + len(set2) - size_intersection)

def flatten_business_rating(line):
	business_id, stderr_rating, user_id_ratings = line[0], line[1][0], line[1][1]
	for user_id, rating in user_id_ratings:
		yield (user_id, (business_id, stderr_rating, rating))

def flatten_business_pair(line):
    user_id = line[0]
    for t1,t2 in line[1]:
        yield ((t1[0], t2[0]), (t1[2] * t2[2], t1[1] * t2[1]))

def get_pearson(line):
    u1, u2 = line[0], line[1][0][0]
    rating1, rating2 = line[1][0][1][1], line[1][1][1]
    stderr_rating1, stderr_rating2 = line[1][0][1][0], line[1][1][0]
    rating1.sort()
    rating2.sort()
    p1 = 0
    p2 = 0
    corated = []
    while p1 < len(rating1) and p2 < len(rating2):
        if rating1[p1][0] < rating2[p2][0]:
            p1 += 1
        elif rating1[p1][0] > rating2[p2][0]:
            p2 += 1
        else:
            corated.append((rating1[p1][1],rating2[p2][1]))
            p1 += 1
            p2 += 1
    if len(corated) < 3:
        return ()
    return ((u1, u2), sum([t[0] * t[1] for t in corated]) / (stderr_rating1 * stderr_rating2))

def item_based(train_review):

	# intermediate: (user_id, (business_id, sqrt(sum of normalized_rating**2), normalized_rating))
	intermediate = train_review \
	            .map(lambda line: (line['business_id'], (line['user_id'], line['stars']))) \
	            .groupByKey() \
	            .mapValues(lambda value: remove_duplicates(list(value))) \
	            .mapValues(lambda value: (sum([t[1] for t in value]) / len(value), value)) \
	            .mapValues(lambda value: [(t[0], t[1] - value[0]) for t in value[1]]) \
	            .mapValues(lambda value: (math.sqrt(sum([t[1] * t[1] for t in value])), value)) \
	            .filter(lambda line: line[1][0] != 0) \
	            .flatMap(flatten_business_rating)

	result = intermediate \
	        .join(intermediate) \
	        .filter(lambda line: line[1][0] != line[1][1]) \
	        .mapValues(lambda value: tuple(sorted(value))) \
	        .groupByKey() \
	        .mapValues(set) \
	        .flatMap(flatten_business_pair) \
	        .groupByKey() \
	        .filter(lambda line: len(list(line[1])) >= 3) \
	        .mapValues(tuple) \
	        .filter(lambda line: line[1][0][1] != 0) \
	        .mapValues(lambda values: sum([t[0] for t in values]) / values[0][1]) \
		    .filter(lambda line: line[1] > 0)

	return result

def user_based(train_review):

	# use minHash and LSH to generate candidate user pairs
	hash_num, b, r = 1600, 800, 2

	params1 = get_params(hash_num)
	params2 = get_params(hash_num)

	business_index = train_review \
		            .map(lambda line: line['business_id']) \
		            .distinct() \
		            .zipWithIndex()
	row_count = business_index.count()

	user_id_business_ids = train_review.map(lambda line: (line['user_id'], line['business_id'])) \
	                        .groupByKey() \
	                        .mapValues(lambda value: set(value))

	user_business_ratings = train_review \
	            .map(lambda line: (line['user_id'], (line['business_id'], line['stars']))) \
	            .groupByKey() \
	            .mapValues(lambda value: remove_duplicates(list(value))) \
	            .mapValues(lambda value: (sum([t[1] for t in value]) / len(value), value)) \
	            .mapValues(lambda value: [(t[0], t[1] - value[0]) for t in value[1]]) \
	            .mapValues(lambda value: (math.sqrt(sum([t[1] * t[1] for t in value])), value)) \
	            .filter(lambda line: line[1][0] != 0) 

	result = train_review \
	            .map(lambda line: (line['business_id'], line['user_id'])) \
	            .join(business_index) \
	            .map(lambda line: (line[1][0], line[1][1])) \
	            .groupByKey() \
	            .mapValues(lambda value: set(value)) \
	            .mapValues(lambda value: generate_sigs(value, hash_num, row_count, params1, params2)) \
	            .flatMap(lambda line: get_bands(line[0], line[1], b, r)) \
	            .groupByKey() \
	            .flatMap(lambda line: get_candidate_pair(set(line[1]))) \
	            .distinct() \
	            .join(user_id_business_ids) \
	            .map(lambda line: (line[1][0], (line[0], line[1][1]))) \
	            .join(user_id_business_ids) \
	            .map(lambda line: ((line[0], line[1][0][0]), (line[1][0][1], line[1][1]))) \
	            .mapValues(get_jaccard_sim) \
	            .filter(lambda line: line[1] >= 0.01) \
	            .map(lambda line: line[0]) \
	            .join(user_business_ratings) \
		        .map(lambda line: (line[1][0], (line[0], line[1][1]))) \
		        .join(user_business_ratings) \
		        .map(get_pearson) \
		        .filter(lambda line: line != () and line[1] > 0)

	return result

def main():
	start = time.time()
	train_file = sys.argv[1]
	model_file = sys.argv[2]
	cf_type = sys.argv[3]

	sc = SparkContext()
	sc.setLogLevel("ERROR")

	train_review = sc.textFile(train_file) \
                .map(lambda line: json.loads(line))

	if cf_type == 'item_based':
		result = item_based(train_review)
		labels = ('b1', 'b2')
	elif cf_type == 'user_based':
		result = user_based(train_review)
		labels = ('u1', 'u2')

	result_ls = result.collect()

	fh = open(model_file, 'w')
	for pair, sim in result_ls:
	    content = {labels[0]: pair[0], labels[1]: pair[1], 'sim': sim}
	    json.dump(content, fh)
	    fh.write('\n')
	fh.close()

	print("Duration: %s" % (time.time() - start))

if __name__ == "__main__":
    main()