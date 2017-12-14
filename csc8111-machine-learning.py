import numpy
import pandas
import random
import seaborn
import matplotlib.pyplot as plot
get_ipython().magic('matplotlib tk')
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load data sets and combine them
train_data_frame = pandas.read_csv('train.csv')
test_data_frame = pandas.read_csv('test.csv')
combined_data_frame = [train_data_frame, test_data_frame]

# Preview Train data frame
train_data_frame.head()

# Preview Test data frame
test_data_frame.head()

# Get some numerical insights from the Train data frame (mean, std, max, min, freq, etc.)
train_data_frame.describe()

# Get some categorical insights from the Train data frame (count, uniqueness, freq, etc.)
train_data_frame.describe(include=['O'])

# Get some numerical insights from the Train data frame (mean, std, max, min, freq, etc.)
test_data_frame.describe()

# Get some categorical insights from the Train data frame (count, uniqueness, freq, etc.)
test_data_frame.describe(include=['O'])

# Analyze correlation between Class and Survival
class_grouping = train_data_frame[['Pclass', 'Survived']] \
    .groupby(['Pclass'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=True)

# Analyze correlation between Sex and Survival
sex_grouping = train_data_frame[['Sex', 'Survived']] \
    .groupby(['Sex'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=True)

# Analyze correlation between the number of Siblings/Spouses and Survival
siblings_spouses_grouping = train_data_frame[['SibSp', 'Survived']] \
    .groupby(['SibSp'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=True)

# Analyze correlation between the number of Parents/Children and Survival
parents_children_grouping = train_data_frame[['Parch', 'Survived']] \
    .groupby(['Parch'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=True)

# Analyze correlation between Embark and Survival
embarked_grouping = train_data_frame[['Embarked', 'Survived']] \
    .groupby(['Embarked'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=True)

seaborn.barplot(x='Embarked', y='Survived', hue='Sex', data=train_data_frame);

# Analyze correlation between Age and Survival
age_chart = seaborn.FacetGrid(train_data_frame, hue='Survived', row='Sex', aspect=4)
age_chart.map(seaborn.kdeplot, 'Age', shade=True)
age_chart.set(xlim=(0, train_data_frame['Age'].max()))
age_chart.add_legend()

# Analyze correlation between Fare, Sex, Embarked and Survival
fare_chart = seaborn.FacetGrid(train_data_frame, row='Embarked', col='Survived', size=2.2, aspect=1.6)
fare_chart.map(seaborn.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=['female', 'male'])
fare_chart.add_legend()

# Convert Sex column to categorical feature
for dataset in combined_data_frame:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

def complete_age_values():
    guess_ages = numpy.zeros((2, 3))

    for dataset in combined_data_frame:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_data_frame = dataset[(dataset['Sex'] == i) & \
                                            (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_guess = guess_data_frame.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int( age_guess / 0.5 + 0.5 ) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), \
                        'Age' ] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

complete_age_values()

def categorize_ages():
    for dataset in combined_data_frame:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']

categorize_ages()

# Check the new categorized data
train_data_frame.head()

# Generate new feature to categorize SibSp and Parch
for dataset in combined_data_frame:
    dataset['IsAlone'] = 0
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Analyze the correlation between the new feature and Survival
alone_grouping = train_data_frame[['IsAlone', 'Survived']] \
    .groupby(['IsAlone'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=True)

# Fill the missing values for the Embarked feature with the most frequent value
most_frequent_embark = train_data_frame['Embarked'].dropna().mode()[0]

for dataset in combined_data_frame:
    dataset['Embarked'] = dataset['Embarked'].fillna(most_frequent_embark)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Fill the missing value on the test data for the Fare feature with the average fare price
average_fare = test_data_frame['Fare'].dropna().median()
test_data_frame['Fare'].fillna(average_fare, inplace=True)

def get_fare_categories():
    for dataset in combined_data_frame:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

get_fare_categories()

# Check again for the new categorized data
train_data_frame.head()

# Cleaning features no longer in use:
# Ticket column (high duplication rate),
# Cabin column (highly incomplete)
# PassengerId column (does not make any contribution)
# Name column (this is non-standard and may cause overfitting)
# SibSp, Parch and FamilSize columns (converted to new feature IsAlone)
def clean_data_frames():
    train_data_frame.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize'], axis=1, inplace=True)
    test_data_frame.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize'], axis=1, inplace=True)
    combined_data_frame = [train_data_frame, test_data_frame]

clean_data_frames()

# Prepare train and test data
x_train = train_data_frame.drop('Survived', axis=1)
y_train = train_data_frame['Survived']
x_test = test_data_frame.drop('PassengerId', axis=1).copy()

shuffle_split = ShuffleSplit(n_splits=20, test_size=.20, random_state=0)

def test_classifier(classifier):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score = cross_val_score(classifier, x_train, y_train, cv=shuffle_split)

    print("Accuracy: %.2f%% (+/- %.2f%%)" % (score.mean() * 100, score.std() * 100))

    return y_pred

# Apply prediction using the classifiers
prediction_lr = test_classifier(LogisticRegression())
prediction_dt = test_classifier(DecisionTreeClassifier(max_depth=10))
prediction_rf = test_classifier(RandomForestClassifier(n_estimators=100))

def save_results(prediction, filename):
    results = pandas.DataFrame({
        'PassengerId': test_data_frame['PassengerId'],
        'Survived': prediction
    })
    results.to_csv(filename, index=False)

# Save the predictions to a file
save_results(prediction_lr, 'results-lr.csv')
save_results(prediction_dt, 'results-dt.csv')
save_results(prediction_rf, 'results-rf.csv')
