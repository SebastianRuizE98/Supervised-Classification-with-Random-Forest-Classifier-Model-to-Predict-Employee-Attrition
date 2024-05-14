import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Load employees data
df = pd.read_csv(r'C:\Users\HP\Desktop\Portafolio data\Bases de datos\HR data\train_data.csv')

# View data
df.head()
df.dtypes
df.keys()
df['Gender'].value_counts()
df['Total Business Value']

# View null values for LastWorkingDate (True or False values for attrition)
df['LastWorkingDate'].notna()

# Create a column for LastWorkingDate bool's
df['Attrition'] = df['LastWorkingDate'].notna()
df['Attrition']
# Drop duplicate employes' ID
df = df.drop_duplicates(subset='Emp_ID')

# Custom Color for graphs
custom_green= (0.0, 1.0, 0.0, 0.4)

# Graph for attrition
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Attrition', data=df, color=custom_green, width=0.3)


# Count attrition values
df['Attrition'].value_counts()

# Percentage if we just gessed 'no' for attrition
(2279 - 102)/2279

# Date data to date dtype
df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], dayfirst=False)
df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], dayfirst=False)
df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], dayfirst=False)

# Emp_ID into object
df['Emp_ID'] = df['Emp_ID'].astype('object')

# Create column for working days
df['DaysWorked'] = pd.NA
df.loc[df['Attrition'] == True, 'DaysWorked'] = df['LastWorkingDate'] - df['Dateofjoining']
df['DaysWorked'].fillna(df['MMM-YY'] - df['Dateofjoining'], inplace=True)

# View dtype object
for column in df.columns:
    if df[column].dtype == object:
        print(f'{column}: {df[column].unique()}')
        print(df[column].value_counts())
        print('_______________________________')

# Attrition to numeric
df['Attrition'] = df['Attrition'].replace({False: 0, True: 1})

# drop useless columns (THIS PART OF THE CODE WAS USED FOR ADJUSTMENTS)
df = df.drop('Emp_ID', axis=1)
df = df.drop('LastWorkingDate', axis=1)
df = df.drop('MMM-YY', axis=1)
df = df.drop('City', axis=1)
df = df.drop('Dateofjoining', axis=1)



# Conversion from non numerical columns to numerical
from sklearn.preprocessing import LabelEncoder
for column in df.columns:
    if not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = LabelEncoder().fit_transform(df[column])

# Put attrition at first
first_col = 'Attrition'
others = [col for col in df.columns if col != first_col]
new_order = [first_col] + others
df = df[new_order]
df['Attrition'].value_counts()
df.head()
df.describe()


# Filter df for True values in attrition
true_attrition = df[df['Attrition'] == 1]


#Filter df for False values in attrition
false_attrition = df[df['Attrition'] == 0]


# X and Y variables definition
X = df.iloc[:, 1:df.shape[1]].values
y = df.iloc[:,0].values

# Split data: 75% training, 25 testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 9, criterion='entropy', random_state=0)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

# Model accuracy score
forest.score(X_train, y_train)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, forest.predict(X_test))

# True Negative, True Positive, False Negative, False positive
TN = cm[0][0]
TP = cm[1][1]

FN = cm[0][1]
FP = cm[1][0]

# Accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
print(cm)
print('Model Testing Accuracy {}'.format(accuracy) )

# Heatmap to ilustrate Confusion matrix
plt.figure(figsize=(9,6))
sns.heatmap(cm, annot=True, fmt='d',linewidths=0.5)

# Model recall
recall = TP/(TP+FN)
print('Recall :', recall)

# Specificity
specificity = TN/(TN+FP)
print('Specificity: ', specificity)

# F1-score
f1_score = 2*(accuracy*recall)/(accuracy+recall)
print('F1-score: ', f1_score)

# Ranking features by importance for the model
importances = forest.feature_importances_
indexes = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indexes[f], importances[indexes[f]]))


# T-student
from scipy.stats import ttest_ind
# List for p-values
p_values = []

# Iteration over each feature
for feature_num in range(X.shape[1]):
    # Calcular el p-value para la prueba t de Student entre las dos clases de 'Attrition'
    t_stat, p_value = ttest_ind(X[y==0, feature_num], X[y==1, feature_num])
    p_values.append(p_value)

sorted_p_values = np.argsort(p_values)
# Print p-values
for i in reversed(sorted_p_values):
    print(f"Feature {i}: p-value = {p_values[i]}")


# GRAPHS
# Countplot for gender attrition
plt.figure(figsize=(6, 4))
plt.xlabel('Gender', fontsize=14, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad= 10)
ax = sns.countplot(x='Gender', data=true_attrition, width=0.4, color=custom_green)
plt.xticks(ticks=[0,1], labels=['Female', 'Male'])

for bar in ax.patches:
    plt.annotate(format(bar.get_height(), '.0f'), 
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                 ha='center', va='bottom', fontsize= 12)




# Attrition by joining designation count
jd_unique = true_attrition['Joining Designation'].value_counts()

# Percentage difference calc
pct_diff = []
prev_value = None
for value in jd_unique:
    if prev_value is not None:
        diff = ((value - prev_value) / prev_value) * 100
        pct_diff.append(diff)
    prev_value = value

print(pct_diff)

                
# Countplot for Joining designation  by attrition

plt.figure(figsize=(6, 4))
plt.xlabel('Joining Designation', fontsize=14, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad= 10)
ax = sns.countplot(x='Joining Designation', data=true_attrition, width=0.4, color= custom_green)

ax.annotate('-38.98%',xy=(0.7,45), fontsize=13)
ax.annotate('-80.56%', xy=(1.7,20), fontsize=13)


for i, rect in enumerate(ax.patches):
    height = rect.get_height()
    plt.plot(rect.get_x() + rect.get_width() / 2, height, 'o', color='red')
    if i < len(ax.patches) - 1:
        next_rect = ax.patches[i+1]
        next_height = next_rect.get_height()
        plt.plot([rect.get_x() + rect.get_width() / 2, next_rect.get_x() + next_rect.get_width() / 2],
                 [height, next_height], color='red', linestyle='-', linewidth=2)
        

plt.show()


# Attrition by Education Level count
education_counts = true_attrition['Education_Level'].value_counts()

# Gent percentage differences for Education Level
ed_pct_diff = []
prev_value2 = None

for value in education_counts:
    if prev_value is not None:
        diff = ((value - prev_value) / prev_value) * 100
        ed_pct_diff.append(diff)
    prev_value = value
print(ed_pct_diff)
    
    
# Countplot for attrition by education level
plt.figure(figsize=(6, 4))
plt.xlabel('Education Level', fontsize= 14, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad= 12)
ax = sns.countplot(x='Education_Level', data=true_attrition, width=0.3, color= custom_green)
plt.xticks(ticks=[0, 1, 2], labels=['College', 'Bacchelor', 'Master'])
ax.annotate('+51.72%', xy=(0.44,42), fontsize=13)
ax.annotate('-34.09%', xy=(1.7,35), fontsize=13)

# Draw a line
for i, rect in enumerate(ax.patches):
    height = rect.get_height()
    plt.plot(rect.get_x() + rect.get_width() / 2, height, 'o', color='red')
    if i < len(ax.patches) - 1:
        next_rect = ax.patches[i+1]
        next_height = next_rect.get_height()
        plt.plot([rect.get_x() + rect.get_width() / 2, next_rect.get_x() + next_rect.get_width() / 2],
                 [height, next_height], color='red', linestyle='-', linewidth=2)

plt.show()

# Stacked barplot for attrition by age
true_attrition['Age'].unique()

f, ax = plt.subplots(figsize=(7, 5))
ax = sns.histplot(x='Age', data=true_attrition, bins=len(true_attrition['Age'].unique()), color=custom_green, linewidth=0.5)
plt.xlabel('Age', fontsize=14, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad=12)
ax.set_xticks([20, 25, 30, 35, 40, 45, 50])

plt.show()

# Boxplots for Dispersion of salaries by attrition status
# Customize a gray color
custom_gray = (0.8, 0.8, 0.8)

colors = {'1': custom_green, '0': custom_gray}

plt.figure(figsize=(6, 4))
sns.boxplot(x='Attrition', y='Salary', data=df, width=0.3, palette=colors)
plt.xlabel('Attrition', fontsize=14, labelpad=10)
plt.ylabel('Salary', fontsize=12, labelpad=12)
plt.show()

# Graph for attrition by designation

jd_unique = true_attrition['Designation'].value_counts()

# Percentage difference calc
pct_diff = []
prev_value = None
for value in jd_unique:
    if prev_value is not None:
        diff = ((value - prev_value) / prev_value) * 100
        pct_diff.append(diff)
    prev_value = value

print(pct_diff)
                
# Countplot for Joining designation  by attrition

plt.figure(figsize=(6, 4))
plt.xlabel('Designation', fontsize=14, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad= 10)
ax = sns.countplot(x='Designation', data=true_attrition, width=0.4, color= custom_green)

ax.annotate('-41.38%',xy=(0.7,45), fontsize=13)
ax.annotate('-73.53%', xy=(1.7,20), fontsize=13)
ax.annotate('-88.88%', xy=(2.5,7), fontsize=13)


for i, rect in enumerate(ax.patches):
    height = rect.get_height()
    plt.plot(rect.get_x() + rect.get_width() / 2, height, 'o', color='red')
    if i < len(ax.patches) - 1:
        next_rect = ax.patches[i+1]
        next_height = next_rect.get_height()
        plt.plot([rect.get_x() + rect.get_width() / 2, next_rect.get_x() + next_rect.get_width() / 2],
                 [height, next_height], color='red', linestyle='-', linewidth=2)
        

plt.show()































