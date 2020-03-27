from pyspark import SparkContext
import json
import math
import sys
import time

def get_cos_sim(profiles):
    p1 = set(profiles[0])
    p2 = set(profiles[1])
    return len(p1 & p2) / (math.sqrt(len(p1)) * math.sqrt(len(p2)))

def main():

	start = time.time()

	test_file = sys.argv[1]
	model_file = sys.argv[2]
	output_file = sys.argv[3]

	sc = SparkContext()
	sc.setLogLevel("ERROR")

	model = sc.textFile(model_file) \
	        .map(lambda line: json.loads(line))

	business_profile = model \
	                .filter(lambda line: 'business_id' in line) \
	                .map(lambda line: (line['business_id'], line['profile']))

	user_profile = model \
	                .filter(lambda line: 'user_id' in line) \
	                .map(lambda line: (line['user_id'], line['profile']))

	test_review = sc.textFile(test_file) \
	        .map(lambda line: json.loads(line)) \
	        .map(lambda line: (line['user_id'], line['business_id']))

	result = test_review \
	        .join(user_profile) \
	        .map(lambda line: (line[1][0], (line[0], line[1][1]))) \
	        .join(business_profile) \
	        .map(lambda line: ((line[1][0][0], line[0]), (line[1][0][1], line[1][1]))) \
	        .mapValues(get_cos_sim) \
	        .filter(lambda line: line[1] >= 0.01)

	result_ls = result.collect()

	fh = open(output_file, 'w')
	for pair, sim in result_ls:
	    content = {'user_id': pair[0], 'business_id': pair[1], 'sim': sim}
	    json.dump(content, fh)
	    fh.write('\n')
	fh.close()

	print("Duration: %s" % (time.time() - start))


if __name__ == "__main__":
    main()