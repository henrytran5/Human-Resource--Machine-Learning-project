●Human Resource dataset
Content and purpose:
 The main goal of this project is to answer the questions that will help a company and get some meaning through the simulated dataset that we have. Our main question that we need an answer to is “Based on various factors from the data, can we predict if an Employee will leave a company or not?”. Other than we have can also add and answer to a few more questions in the future. After we get a decision on which employees are at risk of leaving the company, this can help the company review further questions in the future like:  
•	Why are our best and most experienced employees leaving prematurely?
•	What are the reasons for employee turnover and the future estimate cost-to-hire for budget purposes? 
These are a few questions that could be answered through more data if provided by the company and through the results of our Model. The objective of this project is to discover trends and predict the reasons due to which an employee is likely to leave a company. We have 10 attributes to build the model with and if required reduce the model capacity for a simplistic approach and solution but not compromising on the model performance. 
Apply statistical models:
The decision-making results from the best Model can be used to answer more questions as mentioned above and could be used to build stronger models in the future which can give more accurate answers. One could ask why do we need a model to predict whether an employee would leave a company or not? Why can’t we just predict it by looking at the data? The significance of a model in this case is that we get a systematic decision. We obviously cannot get a correct answer by just looking at the data manually. It would be more of just a guess with lesser accuracy. Human decisions are subjective and might come along with a lot of biases whereas Model decisions are objective which give us decisions with greater accuracy and better results.
About dataset
We are using a dataset related to Human Race Analytics. The objective of this project is to determine why best and most employees are leaving prematurely? There are various factors that can cause an employee to leave. The goal is to also predict which other valuable employee could be the next one to leave based on a classifier that we intend to develop by the end of this project. 
Dataset which will be used for this project is in the link. 
The dataset contains 14999 rows and 10 columns. 
Columns are satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident,left, promotion_last_5years,Department,salary
satisfaction_level: On a scale of 0-1, satisfaction_level gives how satisfied the employee is. 0 is the lowest and 1 is the highest level of satisfaction.
last_evaluation: In years, this value gives time when the last performance evaluation of the employee was done.
number_project: This gives the total number of projects that have been completed by the employee.
average_montly_hours: This gives the number of hours put in by the employee on monthly basis.
time_spend_company:  This gives the number of years spent in the company by the employee.
Work_accident: This gives the count of work related accidents sustained by the employee.
Left: This is provides information whether the employee has left the workplace of not (0 or 1) factor.
promotion_last_5years: Tells us whether the employee has been promoted in last five years or not 
Department: Tells us in what department the employee works for
Salary: Indicator of relative salary (Low or Medium or High)

Variable Name	Datatype
satisfaction_level	Numeric
last_evaluation	Numeric
number_project	Numeric
average_montly_hours	Numeric
time_spend_company	Numeric
Work_accident	Numeric
Left (target)	Numeric
promotion_last_5years	Numeric
sales	Categorical
salary	Categorical

-  Collecting, Combining, Storing, Cleaning, ELT process, Understanding, Analyzing and Predicting 
Collecting, Combining, Storing: from many source complex initial dataset, we will collect, combine and store dataset completely to prepare analytics process
Cleaning: Use software to clean dataset as waiver missing values, duplicate data, wrong data, add more values necessary, the process of fixing or removing incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data within a dataset, etc.
ELT process: Extract- Load-Transform dataset. 
Extract: After collect raw dataset from many source locations, we will extract data as dataset can consist of many data types and come from virtually any structured or unstructured source
Load: the transformed data is moved from the staging area into a data storage area, or between many storing software
Transform: data is to change values to new values for the purport of analytics process. 
•	Filtering, cleansing, de-duplicating, validating and authenticating the data.
•	Performing calculations, translations, data analysis or summaries based on the raw data. 
•	Removing, encrypting, hiding, or otherwise protecting data.
•	Formatting the data into tables or joined tables based on the schema deployed
          -  Find outliers, anomaly detection, missing values, duplicate values, histogram, bar charts, line charts.
         -  Find accuracies, AUC values, ROC curves based on the target attribute to making decisions.

          - Building, developing, and maintaining statistical models for whole dataset in R and Python language
Apply Statistical model and Machine Learning:
- Using statistical models as Linear Regression, Decision Tree, Random Forest, SVM, KNN, Neural Net, Logistics, Bayes Net, K Means, Hierarchical cluster based on 9 attributes and 1 target attribute (Left) with the classification goal is to predict if the employee will be leave a company or not? (outcome target, variable y).
