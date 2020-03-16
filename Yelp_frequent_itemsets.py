from pyspark import SparkContext
import itertools
import sys
import time


def get_bucket_id(pair, bucket_size):
    return (hash(pair[0]) + hash(pair[1])) % bucket_size


def count_itemsets(baskets, itemsets):
    counters = dict()
    for basket in baskets:
        for itemset in itemsets:
            if set(itemset).issubset(basket):
                counters[itemset] = counters.get(itemset, 0) + 1
    return counters


def get_freq_itemsets(counters, support):
    result = []
    for itemset, count in counters.items():
        if count >= support:
            result.append(itemset)
    return result


def generate_candidates_from_lastpass(freq_itemsubsets):
    candidate_itemsets = []
    size = len(freq_itemsubsets[0]) + 1
    if size == 2:
        candidate_itemsets = list(
            itertools.combinations(
                [item for itemsubset in freq_itemsubsets for item in itemsubset], 2
            )
        )
    else:
        n = len(freq_itemsubsets)
        for i in range(n - 1):
            for j in range(i + 1, n):
                subset1 = freq_itemsubsets[i]
                subset2 = freq_itemsubsets[j]
                if subset1[:-1] == subset2[:-1]:
                    candidate = list(subset1[:-1])
                    candidate.extend(sorted([subset1[-1], subset2[-1]]))

                    intermediate_subsets = itertools.combinations(candidate, size - 1)
                    for intermediate_subset in intermediate_subsets:
                        if intermediate_subset not in freq_itemsubsets:
                            candidate = None
                            break

                    if candidate:
                        candidate_itemsets.append(tuple(candidate))

    return candidate_itemsets


def pcy(baskets, support):
    bucket_size = 10**7
    counters = {}
    bucket_table = [0] * bucket_size
    all_freq_itemsets = {}

    # k = 1
    for basket in baskets:
        for item in basket:
            counters[tuple([item])] = counters.get(tuple([item]), 0) + 1

        for pair in itertools.combinations(basket, 2):
            bucket_id = get_bucket_id(pair, bucket_size)
            bucket_table[bucket_id] += 1

    freq_items = sorted(get_freq_itemsets(counters, support))
    if len(freq_items) == 0:
        return []
    all_freq_itemsets[1] = freq_items

    # k = 2
    candidate_pairs = generate_candidates_from_lastpass(freq_items)
    candidate_pairs = [
        pair
        for pair in candidate_pairs
        if bucket_table[get_bucket_id(pair, bucket_size)] >= support
    ]
    del bucket_table
    if len(candidate_pairs) == 0:
        return all_freq_itemsets.values()

    counters = count_itemsets(baskets, candidate_pairs)
    freq_pairs = get_freq_itemsets(counters, support)
    if len(freq_pairs) == 0:
        return all_freq_itemsets.values()
    all_freq_itemsets[2] = freq_pairs

    k = 3
    while True:
        candidate_itemsets = generate_candidates_from_lastpass(all_freq_itemsets[k - 1])
        if len(candidate_itemsets) == 0:
            break

        counters = count_itemsets(baskets, candidate_itemsets)
        freq_itemsets = get_freq_itemsets(counters, support)
        if len(freq_itemsets) == 0:
            break
        else:
            all_freq_itemsets[k] = freq_itemsets
            k += 1

    return all_freq_itemsets.values()


def construct_baskets(sc, input_file, k):
    baskets = (
        sc.textFile(input_file)
        .map(lambda line: line.split(","))
        .filter(lambda line: line[0] != "user_id")
    )


    baskets = baskets.groupByKey() \
            .map(lambda line: (line[0], list(set(line[1])))) \
            .filter(lambda line: len(line[1]) > k)  \
            .map(lambda line: sorted(line[1]))
    return baskets

def son2(k, baskets, candidates):
    if k == 1:
        for basket in baskets:
            for item in basket:
                item = tuple([item])
                if item in candidates:
                    yield(item, 1)
    else:
        for basket in baskets:
            for candidate in candidates:
                if set(candidate).issubset(basket):
                    yield(candidate, 1)

def get_correct_format(line):
    if len(line[0]) == 1:
        to_write = (str(line).strip('[]').replace(' (', '(').replace(',)', ')'))
    else:
        to_write = (str(line).strip('[]').replace(' (', '('))
    return to_write

def main():
    sc = SparkContext()
    sc.setLogLevel("WARN")

    k = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    start = time.time()

    baskets = construct_baskets(sc, input_file, k)

    # son pass 1

    n = baskets.count()
    support_table = dict(
        baskets.mapPartitionsWithIndex(
            lambda chunk_id, chunk: [(chunk_id, len(list(chunk)) / n * support)]
        ).collect()
    )
    son1_result = (
        baskets.mapPartitionsWithIndex(
            lambda chunk_id, chunk: pcy(list(chunk), support_table[chunk_id])
        )
        .flatMap(lambda x: [(i, 1) for i in x])
        .groupByKey()
        .map(lambda x: (len(x[0]), x[0]))
        .groupByKey()
        .mapValues(lambda x: sorted(x))
        .sortByKey()
        .collect()
    )

    # son1_result format: [[('100'), ...], [('100','101'), ...], ...]
    all_candidates = []
    for size, itemsets in son1_result:
        all_candidates.append(list(itemsets))

    # son pass 2

    k = 1
    all_freq_itemsets = []
    for k in range(1, len(all_candidates) + 1):
        son2_result = baskets.mapPartitions(lambda chunk: son2(k, chunk, all_candidates[k - 1])) \
                        .reduceByKey(lambda x, y: x + y) \
                        .filter(lambda line: line[1] >= support) \
                        .map(lambda line: line[0])
        freq_itemsets = sorted(son2_result.collect())
        if len(freq_itemsets) == 0:
            break
        all_freq_itemsets.append(freq_itemsets)

    fh = open(output_file, 'w')
    fh.write('Candidates:\n')
    for line in all_candidates:
        fh.write(get_correct_format(line))
        fh.write('\n\n')
    fh.write('Frequent Itemsets:\n')
    for line in all_freq_itemsets:
        fh.write(get_correct_format(line))
        fh.write('\n\n')
    fh.close()

    print("Duration: %s" % (time.time() - start))

if __name__ == "__main__":
    main()
