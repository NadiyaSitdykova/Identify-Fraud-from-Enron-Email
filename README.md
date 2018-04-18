# P5: Identify Fraud from Enron Email
Enron fraud detection using financial data and emails (Udacity Data Analyst Nanodegree project for Machine Learning course).

### Short questions

__*Question 1: Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?*__

Enron Corporation was an American energy, commodities, and services company. Fortune named Enron "America's Most Innovative Company" for six consecutive years, but at the end of 2001, it was revealed that its reported financial condition was sustained by institutionalized, systematic, and creatively planned accounting fraud, known since as the Enron scandal. During Federal Investigation, the thousands of records (e-mail & financial data) was publicly made. This data was named "Enron corpus". 

The goal of this project is to construct predictive model that could identify individual as a participant of fraud -- Person of Interest(POI), based on financial and email data from Enron corpus. We will use different machine learning techniques to accomplish this goal.

The dataset contains 146 records with 1 label (POI), 14 financial feartures and 6 email features. There are 18 records which are labeled as POIs. The table below describes the number of records with missing values by features.

| Feature                 | Number of records with NaNs |
|-------------------------|-----------------------------|
|POI                      |0                            |
|Bonus                    |64                           |
|Deferral payments        |107                          |
|Deferred income          |97                           |
|Director fees            |129                          |
|Exercised stock options  |44                           |
|Expenses                 |51                           |
|Loan advances            |142                          |
|Long term incentive      |80                           |
|Other                    |53                           |
|Restricted stock         |36                           |
|Restricted stock deferred|128                          |
|Salary                   |51                           |
|Total payments           |21                           |
|Total stock value        |20                           |
|From messages            |60                           |
|From poi to this person  |60                           |
|From this person to poi  |60                           |
|Shared receipt with poi  |60                           |
|To messages              |60                           |

After further exploration I decided to remove three records:

1) `LOCKHART EUGENE E` as it's meaningless to process data containing only NaNs
2) `TOTAL` as it was revealed as an outlier, because it contained summed values for all financial data points
3) `THE TRAVEL AGENCY IN THE PARK` as in seems to be a data-enry error and not an individual

Other outliers wasn't removed as they contained crucial information. Turns out that individuals with significantly high salary and stock options was POIs so we wouldn't like to miss this piece of information, it could be important for our predictive model.

__*Question 2: What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.*__

I engineered three new features: "fraction of emails to poi", "fraction of emails from poi", "fraction of emails with shared receipt with poi". I believe that it's more accurate to use this features, rather than absolute amount of messages to/from/with poi. I suspected that poi would communicate with poi more frequently, than not-poi, so I left all three engineered features in my final feature list. Also I used feature "exercised stock option". This features make sense, as poi supposed to get a lot of money from their fraud exactly by exercising stock option. But actually I got this feature using SelectKBest with k=1. So I had 4 features in my final list and I filled missing values with medians for them. I ended up with that list of features, because they had sense for me and I got better results on them. Then I scaled features via robust scale (appropriate for features with outliers), because I was going to try logistic regression and this algorithm needs features to be scaled. I tried decision tree as well on scaled features, although for this algorithm it wasn't necessary, but it didn't hurt either.

__*Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?*__

I ended up using Logistic Regression (tuned). Also I've tried Naive Bayes, Decision Tree and SVM. Here is a table with two metrics averaged by 1000 differently chosen test dataset (I used StratifiedShuffleSplit with 1000 splits) for algorithms that I've tried: 

| Algorithm                | Recall       | Precision |
|------------------------- |--------------|-----------|
|Naive Bayes               |0.33          |0.46       |
|Logistic Regression       |0.28          |0.72       |
|Tuned Logistic Regression |0.99          |0.32       |
|Decision Tree             |0.21          |0.27       |
|Tuned Decision Tree       |0.67          |0.42       |
|SVM                       |0.27          |0.50       |
|Tuned SVM                 |0.32          |0.28       |

__*Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?*__

Performance of the same algorithm can vary drasticly depending on it's parameters. In order to tune (choosing a set of optimal parameters of an algorithm), we need to establish what is optimal for us (e.g. loss function, that we need to minimize). When I tuned algorithms, firstly I used grid search to find parameters, that would maximize accuracy. This step provided me with a algorithms that satisfied the requirement of project: precision and recall higher than 0.3. Then I manually tuned parameters in order to maximize recall, but without decreasing precison lower than 0.3. For logistic regresssion I tuned tolerance and C, for decision tree -- min samples split, for SVM -- C, gamma and chose between linear and rbf kernels. For all algorithms I used parameter class_weight = 'balanced', because our classes have very different sizes and algorithms are showing better results when classes are balanced.

__*Question 5: What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?*__

Ideally validation is conducted on special validation dataset from the same distribution as train and test data. But in our case dataset is very small, so we can't afford to held back data for validation. If we tune our algorithm on one specific test dataset, we can make a classic mistake of overfitting. We would get algorithm that shows high performance on one particular test dataset, but it's performance on other data could be much worse. In order to more objectively validate algorithm performance without validation dataset, we could use stratified cross-validation. This technique generates asked number of different splits of data on train and test datasets in such manner, that proportion of classes is the same in train and test datasets in each split. I useds split with test_size = 10% of all dataset (default value in StrifiedShuffleSplit in sklearn).

__*Question 6: Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.*__ 

As we can see in a table above, after tuning Logistic Regression showed average value of recall 0.99 and average value of precision 0.32. Recall can be explained as probability that our classifier will identify POI as POI, so we can say our algorithm "catches" 99% of all real POIs. Presision can be explained as probability that identified by our classifer as POI persion is actually POI, so we can say our algorithm wrongfully "catches" 78% of non-POIs. Although classifier with such precision value can be considered as bad for different problems, in our case high recall is much more important. Most straightforward application of our classifier is the narrowing circle of suspects, and our algorithm allows to exclude 32% of employees, which is not bad. Then we can interrogate suspects that left and clear innocent people, but at the same time we can be sure for 99% that real POI won't escape the justice, which is very good.


