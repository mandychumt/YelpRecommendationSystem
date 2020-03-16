from pyspark import SparkContext
import json
sc = SparkContext()

business_id_NV = sc.textFile('business_small.json') \
                .map(lambda line: json.loads(line)) \
                .filter(lambda line: line['state'] == 'NV') \
                .map(lambda line: [line['business_id'], None])

reviews = sc.textFile('review_small.json') \
            .map(lambda line: json.loads(line)) \
            .map(lambda line: [line['business_id'], line['user_id']])

user_id_business_id = business_id_NV.join(reviews) \
                    .map(lambda line: line[1][1] + ',' + line[0])

fh = open('user_business_s.csv', 'w')
fh.write('user_id,business_id')
for line in user_id_business_id.collect():
    fh.write('\n')
    fh.write(line)
fh.close()