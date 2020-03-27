from pyspark import SparkContext
import json
import sys
import time
import statistics

def remove_duplicates(values):
    length = len(values)
    if len(set([t[0] for t in values])) == length:
        return values
    
    values.sort()
    result = []
    l = 0
    r = 1
    while r <= length:
        if r == length or values[l][0] != values[r][0]:
            result.append((values[l][0], statistics.mean([t[1] for t in values[l : r]])))
            l = r
        r += 1
    return result

def flatten_b_id(line):
    user_id, business_id = line[0], line[1][1]
    for b_id, rating in line[1][0]:
        yield (tuple(sorted((business_id, b_id))), (user_id, business_id, rating))

def get_neighbors(value):
    value = list(value)
    value.sort(reverse = True)
    neighbors = []
    for t in value:
        if t[0] > 0.1:
            neighbors.append(t)
    return neighbors

def item_based(sc, train_file, N, test_file, model_file):

	train_review = sc.textFile(train_file) \
		        .map(lambda line: json.loads(line)) \
		        .map(lambda line: ((line['business_id'], line['user_id']), line['stars'])) \
		        .groupByKey() \
		        .mapValues(lambda value: sum(list(value)) / len(list(value))) \
		        .map(lambda line: (line[0][1], (line[0][0], line[1]))) \
		        .groupByKey() \
		        .mapValues(list)

	test_review = sc.textFile(test_file) \
	        .map(lambda line: json.loads(line)) \
	        .map(lambda line: (line['user_id'], line['business_id']))

	model = sc.textFile(model_file) \
	        .map(lambda line: json.loads(line)) \
	        .map(lambda line: ((line['b1'], line['b2']), line['sim'])) 

	result = train_review \
	        .join(test_review) \
	        .flatMap(flatten_b_id) \
	        .join(model) \
	        .map(lambda line: (line[1][0][:2], (line[1][1], line[1][0][2]))) \
	        .groupByKey() \
	        .mapValues(lambda value: tuple(sorted(value, reverse=True)[:N])) \
            .filter(lambda line: line[1][0][0] > 0.01) \
	        .mapValues(lambda value: sum([t[0] * t[1] for t in value]) / sum([t[0] for t in value]))
	return result

def user_based(sc, train_file, N, test_file, model_file):

	# b_id: u_id
	test_pair = sc.textFile(test_file) \
		        .map(lambda line: json.loads(line)) \
		        .map(lambda line: (line['user_id'], line['business_id']))

    # model: business, [(neighgor1, weight) ...]
	model = sc.textFile(model_file) \
	        .map(lambda line: json.loads(line)) \
	        .flatMap(lambda line: [(line['u1'], (line['sim'], line['u2'])), (line['u2'], (line['sim'], line['u1']))]) \
	        .groupByKey() \
	        .mapValues(get_neighbors)

	train_review = sc.textFile(train_file) \
	        .map(lambda line: json.loads(line)) \
	        .map(lambda line: (line['user_id'], (line['business_id'], line['stars']))) \
	        .groupByKey() \
	        .mapValues(lambda values: remove_duplicates(list(values))) \
	        .mapValues(lambda value: (sum([t[1] for t in value]) / len(value), value)) 
	
	# avg_rating: (id, average rating)
	avg_rating = train_review.map(lambda line: (line[0], line[1][0]))

	# train_reivew: (id: (paired_id1, normalized_ratings1), ...)
	train_review = train_review \
	            .mapValues(lambda value: [(t[0], t[1] - value[0]) for t in value[1]]) \
	            .flatMap(lambda line: [((line[0], ID), nor_rating) for ID, nor_rating in line[1]])

	# result: ((user_id, business_id), predicted rating)
	result = test_pair \
	        .join(model) \
	        .flatMap(lambda line: [((ID, line[1][0]), (line[0], weight)) for weight, ID in line[1][1]]) \
	        .join(train_review) \
	        .map(lambda line: ((line[1][0][0], line[0][1]), (line[1][1], line[1][0][1]))) \
	        .groupByKey() \
	        .mapValues(lambda value: sorted(list(value), key=lambda t: t[1], reverse=True)[:N]) \
	        .mapValues(lambda value: sum([t[0] * t[1] for t in value]) / sum([t[1] for t in value])) \
	        .map(lambda line: (line[0][0], (line[0][1], line[1]))) \
	        .join(avg_rating) \
	        .map(lambda line: ((line[0], line[1][0][0]), line[1][0][1] + line[1][1]))
	
	return result

def main():

	start = time.time()

	train_file = sys.argv[1]
	test_file = sys.argv[2]
	model_file = sys.argv[3]
	output_file = sys.argv[4]
	cf_type = sys.argv[5]
	N = 200

	sc = SparkContext()
	sc.setLogLevel("ERROR")

	if cf_type == 'item_based':
		result = item_based(sc, train_file, N, test_file, model_file)
	elif cf_type == 'user_based':
		result = user_based(sc, train_file, N, test_file, model_file)
	
	result_ls = result.collect()
	fh = open(output_file, 'w')
	for pair, stars in result_ls:
	    content = {'user_id': pair[0], 'business_id': pair[1], 'stars': stars}
	    json.dump(content, fh)
	    fh.write('\n')
	fh.close()

	print("Duration: %s" % (time.time() - start))

if __name__ == "__main__":
    main()