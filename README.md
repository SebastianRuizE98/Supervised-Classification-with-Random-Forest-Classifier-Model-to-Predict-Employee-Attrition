# Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition
Sebastián Ruiz Echegaray

**Supervised Classification with Random Forest Classifier Model to Predict Employee Attrition**

**Objective:** The objective of this project is to develop an algorithm comprising a classification model to predict whether an employee will resign or remain employed within the company.

1. **Intro**

Artificial intelligence development has been showing its usefulness in many corporate fields as for example, Human Resources. It can provide tools to optimize decision making processes and management of HR. The aim of this project is to develop an algorithm to predict employee attrition and analyze further decisions regarding HR management.

Random Forest is a machine learning technique that can be used for classification or regression. In this case, it is used for classification purposes. It operates by constructing multiple decision trees at training time. In this classification model, the output is the class selected by most trees. The advantages of implementing a Random Forest Model for these purposes are mainly a controlled variance and correction of overfitting among decision trees.

As employee attrition is a multicausal phenomenon, a Random Forest Model fit the features for this analysis.

1. **Data**

The data was downloaded from [Predicting Employee Attrition (kaggle.com)](https://www.kaggle.com/datasets/pavan9065/predicting-employee-attrition/data?select=train_data.csv). It consisted of a data frame with variables related to employee attrition, for example, age, gender, education level, salary, joining designation, designation, total business value and quarterly ranking. Another variable relevant for this analysis was days worked, but it was calculated from last day worked and first working date, these variables were not relevant to train the model. There were also other variables that were not considered for the analysis and training because of its lack of relevance in this case.

1. **Hyperparameters**

As mentioned earlier, in the context of this study on employee attrition prediction, a Random Forest model, a widely used machine learning technique known for its robustness in classification and regression, was applied. The model was configured with specific hyperparameters to optimize its performance in the task at hand. Specifically, the number of trees (n_estimators) was set to 9, allowing for an appropriate balance between model complexity and computational efficiency. Additionally, the entropy criterion (criterion='entropy') was employed, which is based on Shannon's uncertainty measure to assess the quality of splits in decision trees. This choice was made with the aim of maximizing information gain at each split, leading to more informative trees and, consequently, a more accurate model in predicting employee turnover. The results obtained with this approach support the Random Forest model's ability to identify relevant patterns in the data and provide precise predictions that can be highly beneficial in human resources.

1. **Analysis**

Data was filtered by the condition if attrition was true or false.

The following graphs show the distribution of the values from the relevant variables for the development of the model and the analysis.

**_Graph 1_**. Magnitude of 'attrition' vs 'no-attrition'
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/Attrition%20graph.png?raw=true" alt="Attrition Graph">
</p>



**_Graph 2_**. Distribution of attrition according to education level
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/ed%20lvl%20graph.png?raw=true" alt="Education Level Graph">
</p>


This graph suggests that college students have a higher probability for attrition.

**_Graph 3_**. Distribution of attrition according to gender
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/gender_graph.png?raw=true" alt="Gender Graph">
</p>



This graph suggests that male employees have a higher probability for attrition.

**_Graph 4_**. Distribution of attrition according to joining designation
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/joining%20designation%20graph.png?raw=true" alt="Joining Designation Graph">
</p>





It is imperative to note the absence of explicit data regarding the values assigned to the joining designations within the dataset. To address this limitation, an analytical approach was adopted, leveraging the frequency distribution of each joining designation categorized by education level. This methodology was predicated on the assumption that the organizational hierarchy of education levels significantly influences the designation of roles upon individuals joining the company. We cannot conclude the presence of a hierarchical structure for the values assigned to each role.

This graph suggests that employees whose joining designation is ‘1’ have the highest probabilities for attrition followed by the ones whose ‘joining designation is ‘2’

**_Graph 5_**. Distribution of attrition according to designation
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/designation_graph.png?raw=true" alt="Designation Graph">
</p>




The 'designation' variable indicates the role designation held by employees as of the date when the data was collected. This graph shows the order according to the probability for attrition.

**_Graph 6_**. Distribution of attrition by age
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/age_graph.png?raw=true" alt="Age Graph">
</p>



**_Graph 7_**. Dispersion of salaries by attrition status
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/salary_graph.png?raw=true" alt="Salary Graph">
</p>



This graph show the dispersion of salaries by attrition status. In this graph, number 0 stands for ‘False’ and number 1 for ‘True’. We can observe that the salaries’ mean for the ‘False’ status (59887.875822729264) is higher than the mean for ‘True’ status (44042.205882352944). The percentage difference between those two means is 26.45889459709708 %. This is a significant difference to consider this factor as a strong predictor for employee attrition.

1. **Discussion**

The model was trained using the 'forest.feature_importance' function to select the most relevant features for training while discarding those that decreased prediction accuracy. Variables were iteratively eliminated one by one, and the model was retrained until maximum achievable accuracy was reached. The achieved prediction accuracy of the model was 96.140%. Additional error metrics were implemented to assess false and true positives. Recall, specificity, and F1 score were employed to validate the model in addition to the calculated accuracy. Recall measures the proportion of actual positives that were correctly identified by the model (47.826%), specificity measures the proportion of actual negatives that were correctly identified (98.080%), and F1 score provides a balance between precision and recall (63.876%).

**_Graph 8_**. Confusion Matrix
<p align="center">
  <img src="https://github.com/SebastianRuizE98/Supervised-Classification-with-Random-Forest-Classifier-Model-to-Predict-Employee-Attrition/blob/main/confusion_matrix.png?raw=true" alt="Confusion Matrix">
</p>



In this graph, number 0 stands for ‘False’ and number 1 for ‘True’. The confusion matrix provides a detailed breakdown of the performance of a binary classification model. In this particular instance, the matrix reveals that the model correctly classified 562 instances as negative (True Negatives) and 11 instances as positive (True Positives). However, the model also misclassified 12 instances of actual positives as negatives (False Negatives) and 11 instances of actual negatives as positives (False Positives).

While the model demonstrates a strong ability to correctly identify both negative and positive instances, the presence of False Negatives and False Positives indicates areas where the model's performance can be improved. False Negatives suggest instances where the model failed to detect positive outcomes, potentially leading to missed opportunities or incorrect decisions. Conversely, False Positives may result in unnecessary actions or resources being allocated.

1. **Conclusion**

In conclusion, the implementation of a Random Forest Classifier model for predicting employee attrition proved to be a valuable approach in this context, shedding light on various factors influencing employee turnover within the organization. Through the analysis of relevant variables such as education level, gender, joining designation, designation, age, and salary, notable insights were gained into the dynamics of attrition patterns.

Among these variables, salary emerged as a particularly strong predictor of attrition, with a significant difference observed between the mean salaries of employees with true and false attrition statuses. This underscores the importance of considering compensation structures and salary levels in understanding and mitigating attrition risks.

Furthermore, the model demonstrated commendable predictive accuracy, achieving an overall accuracy rate of 96.140%. However, a detailed examination of the confusion matrix revealed certain limitations, notably the model's tendency to perform better in identifying True Negatives compared to other categories. In the context of employee attrition prediction, True Negatives happen to illustrate employees who will not resign. This emphasizes the need for continued refinement and optimization of the model to minimize mainly false positives, thus enhancing its practical utility in HR management.

Despite these limitations, the model's ability to accurately classify True Negatives (562 instances) underscores its value in identifying employees who are unlikely to leave the organization, thereby facilitating targeted retention efforts. Conversely, the misclassification of False Negatives (6 instances) and False Positives (11 instances) highlights areas where further improvements can be made to enhance predictive accuracy and reduce the likelihood of erroneous predictions.

Practically, the insights generated from this analysis can inform HR decision-making processes, enabling proactive interventions to retain valuable talent and address attrition risks effectively. By leveraging advanced machine learning techniques and comprehensive data analysis, organizations can gain valuable insights into employee behavior and drive strategic HR initiatives to foster a positive work environment and enhance employee satisfaction and retention.
