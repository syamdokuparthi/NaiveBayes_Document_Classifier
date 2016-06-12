# NaiveBayes_Document_Classifier
In this project, Naive Bayes document classifier implemeneted and applied to the 20 newsgroups dataset to Predict which newsgroup a given document was posted to  Maximum Likelihood Estimation (MLE), Maximum a posteriori (MAP) are estimated and Naive Bayes Classifier is built and the test data is classified in to 20 news groups.  Misclassification is identified with the confusion Matrix.

Naive Bayes Classifier:

Naive Bayes classifiers is a simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. It performs text categorization, the problem of judging documents as belonging to one category or the other (such as spam or legitimate, sports or politics, etc.) with word frequencies as the features.

Abstractly, naive Bayes is a conditional probability model, given a problem instance to be classified, represented by a vector representing some n features (dependent variables), it assigns to instance probabilities for each of k possible outcomes or classes. The main drawback of this algorithm is that the far-reaching independence assumptions are often inaccurate and fails to produce a good estimate for the correct class probabilities. Naive Bayes classifier will make the correct MAP decision rule classification so long as the correct class is more probable than any other class. This is true regardless of whether the probability estimate is slightly, or even grossly inaccurate. In this manner, the overall classifier can be robust enough to ignore serious deficiencies in its underlying naive probability model.



Algorithm of Code working:

1.	Load the train.label, train.data, vocabulary.txt, test.label, test.data files using matlab script generator for file loading and store in the variables created with file name.

2.	From train_lable compute the count of the documents appeared by the repeated values of the document numbers for 20 categories individually. Then compute the probability of the document occurrence in category by dividing the count values with the total length of the train_label variable which is MLE.

3.	 From the train.data, generate count_matrix by updating the count values which constitutes of segmenting the document IDS from each category to the corresponding group(category) using the train_label content as the row value.

4.	That is by the above step we will be generating 20 rows which represents 20 categories and each row constitutes of the doc_id as elements, ie columns of the matrix.

5.	Now compute the conditional_probability_matrix which is the MAP calculation by the mathematical formula below by taking beta=( α-1) =1/vocabulary count

     P(Xi  | Yk ) = ((count of Xi in Yk )+(α - 1)) / ((total words in Yk )+( α - 1)*length of vocab list).

6.	Update the test_count_matrix from the test_matrix with category value, Document id from the count value present in test.data.

7.	Compute with the mathematical formula

      Ynew = test_count_matrix*log2(transpose(conditional_probability_matrix))

8.	Complete the text  classification with the mathematical formula and then finding the max(Yfinal)
       
      Yfinal = (Ynew+log2(doc_prob_transpose))

9.	Since the dimensions of Ynew matrix is different from the log2(doc_prob_transpose) the values log2(doc_prob_transpose) is added to each and every row of Ynew.

10.	Now in category=  max(Yfinal), we will get the category values of the documents which we categorized with the test.data. 
11.	Now we will calculate the accuracy of our algorithm by comparing the values of the predicted categorization with the actual category values of test.label and computing correct/(correct +wrong) count. That is correctly classified and total classified values.

12.	Now change our assumption of beta=1/”vocabulary count” to beta=values between .0001-1, which I considered here as 1000 equidistant values.

13.	Also compute the accuracy for the different beta values and store in the accuracy array and plot the graph with semilogx(beta_count,accuracy).

14.	Compute the confusion_matrix with the predefined function confusionmat using test_label, predicted category.

15.	To find the misclassified distribution of categories with our algorithm compute the % of elements to the total elements in that row and repeat the same for all the 20 rows. By the % distribution in each row we can understand the misclassification of the documents in that category.

16.	Now finding the high frequency words is done by using the MAP values. That is using the conditional probability concept i.e with the probability of a word occurrence in all categories and then with maximum MAP value of the divided by the total conditional probability of occurrence of that word in all the categories gives a value which can be sorted in descending order to find the ranking of the words. Here sorting is done with sort function in descending order and retrieving the index of the sort list to find the words from the vocabulary list with the index.





Accuracy obtained:

The accuracy obtained with the Naïve Bayes classification algorithm is 78.520986009327120
Considering beta=( α-1) =1/vocabulary count.
But on changing our assumption of changing beta value in the calculation of MAP between 
.0001-1, I observed the value of accuracy is varied between 78%-81%.




Which option works well:

a.	The file loading is done with the matlab generated file loading scrip rather than using dlmread command since I observed loading the file with dlmread command is taking long time due the misalignment of the space(“ “) in the documents provided. Even using the dlmread command by keeping space as delimiter I observed the runtime is a bit long than the matlab file loading script hence I used file loading script to decrease the program execution time.
 
b.	The initial computation of the document probability is calculated by traversing the whole document only once with an efficient algorithm of counting the documents which greatly reduced the program runtime. The logic worked efficiently since the train.label file is already in sorted order and facilitated to count the values in incrementing order easily.

c.	During the computation of the Yfinal the doc_prob_transpose of dimension 1x20 has to be generated 7505 time to match with the Ynew dimensions 7505x20. Generation of doc_prob_transpose with repmat function and then adding with Ynew is taking more execution time and to eliminate that issue I added doc_prob_transpose to every row  Ynew which boosted the execution time of program.



It is difficult to accurately estimate the parameter of this model since the given document count 1000 with each 1000 words will provide a good classification when vocabulary list is less since from the train data we could see that the given 50,000 vocabulary is proportional to the train vocabulary but the training document count is 11000 which is much higher to given document count leading to 78% accuracy. In short the given documents count is much less with which the classification is difficult to attain classification with a decent accuracy because of the high vocabulary provided.

The overall test accuracy obtained with the Naïve Bayes classification algorithm is 78.520986009327120 Considering beta =( α-1) =1/vocabulary count.

Confusion matrix is computed with function “confusionmat” using test_label, predicted category list

 

Yes the algorithm confuses to classify the document categories correctly more often other than others for some categories. This can be observed from the percentage misclassified distribution of categories with our algorithm by computing the % of elements to the total elements in that row of the confusion matrix. By the % distribution in each row we can understand the misclassification of the documents in that category.
 
From the above calculation we can see that the category 1 (alt.ateism), category 16 (soc.religion.christian) and category 20 (talk.religion.misc) documents got misplaced with each other since all these categories are related fields.
Also we could see that a greater misclassification is seen for categories 4, 5 which relates to hardware for a computer. Likewise categories 2,6,13 related to similar electronics group and produces a wrong classification of documents, Similarly 17, 18, 19 groups which relates politics.


By changing beta value in the calculation of MAP between .0001-1, I observed the value of accuracy is varied between 78%-81%. I used linspace command to divide the range between .0001-1 in to 1000 values and plotted the semilogx graph beta_count, accuracy values stored in arrays.
I could see that the accuracy is dropping for small and large values of “beta” because when the beta values are small the classifier MAP value generated will be very less value leading to negligible difference between MAP values making the classifier very difficult to differentiate the values accurately, i.e the classifier will be acting very strict in classifying words and will not even categorize the word belonging to that actual group.
When the beta values are large the classifier will be acting very liberally in the classification of words leading to greater misclassification. i.e the word not belonging to that specific group will be even allowed to fall in to that group leading to mismatch of the word classification. 


With the P(X/Y) value we will get the probability of occurrence of a word in all the categories. With that value we cannot rank the words since P(X/Y) value will be more for the words like “”the, “of” , “I”, “with” etc in all the rows of a word index column which will not provide considerable information about the document. This domination of unimportant words can be nullified by dividing these probabilities with a value that related the maximum occurrence of the stop words. Since the conditional probability for all these stop words is also very high I planned to divide the max conditional probability of the word by the sum of all conditional probability of that specific word by which the resulting value will be a low value which can be taken as a parameter to rank the words.
Hence by computing the same value for all the 61188 words in 20 categories and sorting the resulting value in descending order, we will get the highest ranking words in order. I have mentioned the 100 words in the list as solution to question 6.


The top 100 words occurred 
'nhl'	'stephanopoulos'	'leafs'	'alomar'	'wolverine'	'crypto'	'lemieux'	'oname'	'rsa'	'ripem'	'athos'	'rbi'	'firearm'	'powerbook'	'pitcher'	'dyer'	'bruins'	'lciii'	'lindros'	'fprintf'	'ahl'	'azerbaijan'	'candida'	'iisi'	'args'	'baerga'	'gilmour'	'gfci'	'clh'	'pitchers'	'clemens'	'dodgers'	'gainey'	'sabretooth'	'liefeld'	'rlk'	'jagr'	'adb'	'hobgoblin'	'hawks'	'crypt'	'anonymity'	'aspi'	'countersteering'	'punisher'	'xfree'	'azerbaijani'	'cipher'	'recchi'	'sdpa'	'oilers'	'soderstrom'	'obp'	'argic'	'libxmu'	'jaeger'	'goalie'	'serdar'	'inning'	'sumgait'	'xmu'	'umu'	'gaza'	'denning'	'ioccc'	'obfuscated'	'rayshade'	'nsmca'	'xdm'	'ranck'	'dineen'	'stderr'	'dpy'	'cardinals'	'homicides'	'orbiter'	'mozumder'	'potvin'	'sandberg'	'uccxkvb'	'imake'	'plaintext'	'whalers'	'moncton'	'mydisplay'	'wip'	'hicnet'	'steveh'	'bontchev'	'karabakh'	'baku'	'canadiens'	'messier'	'bure'	'bikers'	'cryptographic'	'mutants'	'keown'	'ssto'	'ashok'




Rather than saying that the algorithm is biased, I would say that the algorithm is trained or developed on this specific set of input words, documents to execute on the similar data set to classify. As time changes the words usage may be modified and new words might replace the old words to describe things and even the words which we use now might also be used to describe another thing which will result in document classification failure and very low accuracies. Hence I would say that the algorithm has to be trained after certain period of time to retain proper text classification and accuracies.  Also machine learning methods will greatly improve and change with time.
