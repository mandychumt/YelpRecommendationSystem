# Yelp Recommendation Systems

----------- Collaborative Filtering Recommendation System -----------

Collaborative filtering recommendation systems are built with train reviews and use the models to predict the ratings for a pair of user and business.

Case 1: Item-based CF recommendation system
A model is built by computing the Pearson correlation for the business pairs that have at least three co-rated users. During the predicting process, the model is used to predict the rating for a given pair of user and business.

Case 2: User-based CF recommendation system with Min-Hash LSH
During the training process, since the number of potential user pairs might be too large to compute, the system combines Min-Hash and LSH algorithms in the user-based CF recommendation system. Then, Pearson correlation is computed for the user pair candidates that have Jaccard similarity >= 0.01 and at least three co-rated businesses. The predicting process is similar to Case 1.

Training command: $ spark-submit CF_train.py <train_file> <model _file> <cf_type>
  
Predicting command: $ spark-submit CF_predict.py <train_file> <test_file> <model_file> <output_file> <cf_type>

<cf_type>: either “item_based” or “user_based”

<output_file>: CF_item_based.predict, CF_user_based.predict


-------------- Content-based Recommendation System --------------

A content-based recommendation system is built by generating profiles from review texts for users and businesses in the train review set. Then, the system/model is used to predict if a user prefers to review a given business, i.e., computing the cosine similarity between the user and item profile vectors.

All the review texts are concatenated for the business as the document and parsing the document. Punctuations, numbers and stopwords, and extremely rare words are removed to reduce the vocabulary size, i.e., the count is less than 0.0001% of the total words. After that, Word importance is measured using TF-IDF, i.e., term frequency * inverse doc frequency and take top 200 words with highest TF-IDF scores to describe the document. At the same time, user profile is developed by aggregating the profiles of the items that the user has reviewed.

During the predicting process, the system estimate if a user would prefer to review a business by computing the cosine distance between the profile vectors.

Training command: $ spark-submit content_based_train.py <train_file> <model_file> <stopwords>
  
Predicting command: $ spark-submit content_based_predict.py <test_file> <model_file> <output_file>

<output_file>: content_based.predict


----------------- Similar Business Pairs -----------------

Min-Hash and Locality Sensitive Hashing algorithms are implemented with Jaccard similarity to find similar business pairs in the train_review file. This system focuses on the 0 or 1 ratings rather than the actual ratings/stars in the reviews. After finding candidate pairs, the system verifies the candidate pairs using their original Jaccard similarity and outputs business pairs whose Jaccard similarity is >= 0.05.

Execution command: $ spark-submit similar_business_pairs.py <input_file> <output_file>

<output_file>: similar_business_pairs


-------------------- Frequent Itemsets --------------------

Used pyspark to implement SON and PCY algorithms to calculate the combinations of frequent businesses (as singletons, pairs, triples, etc.) that are qualified as frequent given a support threshold.

Execution command: $ spark-submit Yelp_frequent_itemsets.py <filter threshold> <support> <input_file_path> <output_file_path>
