# Information Retrieval
Implemenation of projects

Spring 2021

Phase 1. : Designing and implementation of an information retrieval system for [movies](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset) dataset

      The main purpose of this phase was to create an information retrieval system which I first preprocess texts using stemming, lemmatization, stop-words. Then, index the documents using positinal-index , dynamic indexing and bigram. After that i implemented some index compression method such as gamma-code, variable-byte and then added spell correction with the use of that bigram index, jaccard and edit distance. Finally, the documents were retrieved using TF-IDF algorithm and highlight the word that had an effect on the scoring algorithm. 
      
      At the end, the efficiency of the system was evaluated using precision, recall, f1, ap, map and ndcg.
       

Phase 2. Machine Learning methods in text-processing
       The main purpose of this phase was to implement different classification and clustering algorithms. At first I preprocess data using TF-IDF algorithm. Then I tested multiple Classification algorithms such as Naive-Bayes, KNN, SVM, Neural-Network and clustering algorithms like K-means, Gaussian-Mixture-Models, Hierarchical-Clustering.
       
       At the end, the efficiency of these algorithms were compared using multiple evaluation metrics such as accuracy, precision, recall, f1 and purity.
       

Phase 3. Recommender System for [academic microsoft](https://academic.microsoft.com/) web pages:
       The main purpose of this phase was to create a recommender system for extracting essential data from academic microsoft web pages. First I implemented a crawler for fetching papers' information. Then I rank the pages by PageRank algorithm and then rank the author by HITS algorithm. Finally, I create a recommender system once with Content-based method and then with Collaberative filtering.
       
       
A simple user interface was implemented as well for easier utilization.
