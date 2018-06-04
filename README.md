# search-basic

Basic Document Retrieval system.

Works on 4 types of queries: (refer the screenshot)
1) single word
2) multi-word (with at least one of the words present) //Union in backend
3) multi-word (will all words present in the document) //Intersection on backend
4) phrase search (all words must be present in the same order)

Things to improve:
1) Currently always calculates TF-IDF index (make it more like normal inverted index which is not recomputed every time)
2) Test on whole set (instead of 100 docs go for all 20K)
3) Use stemmming and lemmatizing functions
4) Try replacing numbers with the tag <NUMBER> (might help in increasing accuracy)
5) Try creating benchmarks. Precision (including P@10), Recall, MAP are vital.

Referred:
1) [Implementing a Search Engine with Ranking in Python](http://aakashjapi.com/fuckin-search-engines-how-do-they-work/)
2) [Working with Text data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
