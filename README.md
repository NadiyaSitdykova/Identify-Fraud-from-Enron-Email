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

I engineered three new features: "fraction of emails to poi", "fraction of emails from poi", "fraction of emails with shared receipt with poi". I believe that it's more accurate to use this features, rather than absolute amount of messages to/from/with poi. I suspected that poi would communicate with poi more frequently, than not-poi, so I left all three engineered features in my final feature list. Also I used features "exercised stock option", "total stock value", "bonus" and "salary". All these features make sense, as poi supposed to get a lot of money from their fraud. But actually I got this 4 features and "fraction to poi" using SelectKBest with k=5. I decided to use k=5, because the 6th feature was "deferred income" and as we can see in table above this feature 97 missing values, which is more than a half of data. I though that it would be unwise to use such feature as well as other that have even smaller score. So I had 7 features in my final list and I filled missing values with medians for them. Then I scaled feature via min-max, because I was going to try logistic regression and this algorithm needs features to be scaled. I tried decision tree as well on scaled features, although for this algorithm it wasn't necessary, but it didn't hurt either.

__*Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?*__

__*Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?*__

__*Question 5: What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?*__

__*Question 6: Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.*__ 


