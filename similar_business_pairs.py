from pyspark import SparkContext
import json
import itertools
import sys
import random
import time

def hash_func(param1, param2, row_count, v):
    return ((param1 * v + param2) % 26189) % row_count

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

def main():
    start = time.time()

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    sc = SparkContext()
    sc.setLogLevel("ERROR")
    sc._conf.set("spark.python.profile", "true")
    
    train_review = sc.textFile(input_file) \
                .map(lambda line: json.loads(line)) 

    user_index = train_review \
                .map(lambda line: line['user_id']) \
                .distinct() \
                .zipWithIndex()
    row_count = user_index.count()

    business_id_user_ids = train_review.map(lambda line: (line['business_id'], line['user_id'])) \
                            .groupByKey() \
                            .mapValues(lambda value: set(value))

    hash_num, b, r = 40, 40, 1
    params1 = get_params(hash_num)
    params2 = get_params(hash_num)

    result = train_review \
            .map(lambda line: (line['user_id'], line['business_id'])) \
            .join(user_index) \
            .map(lambda line: (line[1][0], line[1][1])) \
            .groupByKey() \
            .mapValues(lambda value: set(value)) \
            .mapValues(lambda value: generate_sigs(value, hash_num, row_count, params1, params2)) \
            .flatMap(lambda line: get_bands(line[0], line[1], b, r)) \
            .groupByKey() \
            .flatMap(lambda line: get_candidate_pair(set(line[1]))) \
            .distinct() \
            .join(business_id_user_ids) \
            .map(lambda line: (line[1][0], (line[0], line[1][1]))) \
            .join(business_id_user_ids) \
            .map(lambda line: ((line[0], line[1][0][0]), (line[1][0][1], line[1][1]))) \
            .mapValues(get_jaccard_sim) \
            .filter(lambda line: line[1] >= 0.05)

    # output result
    result_ls = result.collect()
    fh = open(output_file, 'w')
    for pair, sim in result_ls:
        json.dump({'b1': pair[0], 'b2': pair[1], 'sim': sim}, fh)
        fh.write('\n')
    fh.close()

    print("Duration: %s" % (time.time() - start))

if __name__ == "__main__":
    main()