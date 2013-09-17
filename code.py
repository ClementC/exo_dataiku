import pandas as pd

data_dir = r'C:\Documents and Settings\Clement\Desktop\Dropbox\personal\exercice_dataiku\us_census_full'

# Get labels
import os

metadata_file = os.path.join(data_dir, 'census_income_metadata.txt')
labels = [e.split(':')[0] for e in open(metadata_file).readlines()[-42:] if not e.startswith('|')] + ['class']      # ignore the "| ignore weight" feature

# Get data
data_file = os.path.join(data_dir, 'census_income_learn.csv')
data = pd.read_csv(data_file, names = labels)

# Describe data very simply (statistical moments for each dimension - mode for nominal data)
for l in data.columns:
    print l
    print data[l].describe()
    print "\n\n"

# Print parameters with missing values and corresponding percentage of missing values
for l in sorted(data.columns):
    for v in set(data[l].values):
        if str(v).find('?') != -1:
            print '{:<40} {}%'.format(l, "{0:.2f}".format(100*float(len(data[l][data[l] == v]))/len(data[l])))
            break

# Number of instances with missing values
records_with_missing_values = len([e for v in data.values if ' ?' in map(str, v)])
print "Number of records with missing values:", records_with_missing_values, "({:.2f}% of total).".format(100*float(records_with_missing_values) / data.shape[0])

# Print distributions for the two different classes
import matplotlib.pyplot as plt

over = data[data['class'] == " 50000+."]
below = data[data['class'] == " - 50000."]

for l in data.columns:
    if data[l].dtype.name != 'object':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        minimum_v = data[l].min()
        maximum_v = data[l].max()
        over[l].hist(color='green', alpha=0.5, bins=50, range = (minimum_v, maximum_v), normed = True)
        below[l].hist(color='red', alpha=0.5, bins=50, range = (minimum_v, maximum_v), normed = True)
        ax.set_xlabel(l)
        ax.set_ylabel("Frequency")
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc="green", alpha = 0.75),
                   plt.Rectangle((0, 0), 1, 1, fc="red", alpha = 0.75),],
                  ['Over 50k$', 'Below 50k$'])
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        possible_values = sorted(list(set(data[l])))
        values = dict(zip(possible_values, [0. for i in range(len(possible_values))]))
        partial_values = (over[l].value_counts()/float(over.shape[0]))
        for v in partial_values.index:
            values[v] = partial_values[v]
        pd.Series(values, index = sorted(values.keys())).plot(kind = 'bar', color='green', alpha=0.5)
        values = dict(zip(possible_values, [0. for i in range(len(possible_values))]))
        partial_values = (below[l].value_counts()/float(below.shape[0]))
        for v in partial_values.index:
            values[v] = partial_values[v]
        pd.Series(values, index = sorted(values.keys())).plot(kind = 'bar', color='red', alpha=0.5)
        ax.set_xlabel(l)
        ax.set_ylabel("Frequency")
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc="green", alpha = 0.75),
                   plt.Rectangle((0, 0), 1, 1, fc="red", alpha = 0.75),],
                  ['Over 50k$', 'Below 50k$'])
        plt.show()

# Typical citizen for each class
print "Typical citizen over 50k$"
for l in over.columns:
    print '\t\t{:<50} {}'.format(l, over[l].median() if over[l].dtype.name != 'object' else over[l].describe()['top'].strip())

print "\n\nTypical citizen below 50k$"
for l in below.columns:
    print '\t\t{:<50} {}'.format(l, below[l].median() if below[l].dtype.name != 'object' else below[l].describe()['top'].strip())

print "\n\n{:<60}{:<50}{}".format("Notable differences", "Over", "Below")
for l in over.columns:
    over_c = str(over[l].median()) if over[l].dtype.name != 'object' else over[l].describe()['top'].strip()
    below_c = str(below[l].median()) if below[l].dtype.name != 'object' else below[l].describe()['top'].strip()
    if over_c != below_c:
        print '\t\t{:<50} {:<40}\t\t{}'.format(l, over_c, below_c)

# Transform categorical data into boolean data
import random

selected_dims = ['age', 'class of worker', 'education', 'wage per hour', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'capital gains', 'capital losses', 'dividends from stocks', 'tax filer stat', 'region of previous residence', 'detailed household summary in household', 'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt', 'num persons worked for employer', 'family members under 18', 'citizenship', 'own business or self employed', "fill inc questionnaire for veteran's admin", 'veterans benefits', 'weeks worked in year', 'year', 'class']
t_data = pd.DataFrame({'zero': pd.Series([0 for i in range(data.shape[0])], index = data.index).to_sparse()})
for l in selected_dims[:-1]:
    if data[l].dtype.name == 'object':
        attributes = list(set(data[l].values))
        for a in attributes:
            t_data[l + '_' + a] = pd.Series([1 if e == a else 0 for e in data[l].values], index = data.index).to_sparse()
    else:
        t_data[l] = pd.Series(data[l].values, index = data.index).to_sparse()
t_data['class'] = pd.Series([1 if e == " 50000+." else -1 for e in data['class'].values], index = data.index).to_sparse()
del t_data['zero']
del data

# Correlate class with variables, print first ten most correlated variables
import math

correlations = sorted([[l, t_data[l].corr(t_data['class'])] for l in t_data.columns if l != 'class'], key = lambda x: math.fabs(x[1]), reverse = True)
for c in correlations[:10]:
    print '{:<60} {}'.format(*c)

# Select most interesting continuous dimensions
interesting_dims = [e[0] for e in correlations if e[0].find('_') == -1][:3] + ['class']

# Subsample "Below" class because of memory problems
a = t_data[t_data['class'] == 1]
b = t_data[t_data['class'] == -1].iloc[random.sample(range(below.shape[0]), over.shape[0])]
t_data = pd.concat([a, b], ignore_index = True)
del a
del b

# 3D plot with the most interesting dimensions
from mpl_toolkits.mplot3d import Axes3D
import random

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dims = [t_data[t_data['class'] == 1][l] for l in interesting_dims[:-1]]
ax.scatter(*dims,
           color = 'green', alpha = 0.75)
dims = [t_data[t_data['class'] == -1][l] for l in interesting_dims[:-1]]
ax.scatter(*dims,
           color = 'red', alpha = 0.75)

ax.set_xlabel(interesting_dims[0])
ax.set_ylabel(interesting_dims[1])
ax.set_zlabel(interesting_dims[2])

ax.legend([plt.Rectangle((0, 0), 1, 1, fc="green", alpha = 0.75),
           plt.Rectangle((0, 0), 1, 1, fc="red", alpha = 0.75),],
          ['Over 50k$', 'Below 50k$'])

plt.show()

# Train linear model
from sklearn import cross_validation, svm           # Use StratifiedKFold
import numpy as np

interesting_dims = [e[0] for e in correlations][:20] + ['class']

# Train with CV (add grid search)
X_over = t_data[t_data['class'] == 1]
X_below = t_data[t_data['class'] == -1]
X = np.vstack([np.array(X_over.iloc[:,:-1]), np.array(X_below.iloc[:,:-1])])
y = np.concatenate([np.array(X_over['class']), np.array(X_below['class'])])

from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(X, y)

# Get performance on test data
testing_data_file = os.path.join(data_dir, 'census_income_test.csv')
testing_data = pd.read_csv(testing_data_file, names = labels)
# Transform testing data
t_data = pd.DataFrame({'zero': pd.Series([0 for i in range(testing_data.shape[0])], index = testing_data.index).to_sparse()})
for l in selected_dims[:-1]:
    if testing_data[l].dtype.name == 'object':
        attributes = list(set(testing_data[l].values))
        for a in attributes:
            t_data[l + '_' + a] = pd.Series([1 if e == a else 0 for e in testing_data[l].values], index = testing_data.index).to_sparse()
    else:
        t_data[l] = pd.Series(testing_data[l].values, index = testing_data.index).to_sparse()
t_data['class'] = pd.Series([1 if e == " 50000+." else -1 for e in testing_data['class'].values], index = testing_data.index).to_sparse()
del t_data['zero']
del testing_data
# Subsample "Below" class because of memory problems
a = t_data[t_data['class'] == 1]
b = t_data[t_data['class'] == -1]
t_data = pd.concat([a, b], ignore_index = True)
del a
del b

X_test = np.array(t_data.iloc[:,:-1])
y_test = np.array(t_data['class'])

del t_data

from sklearn.metrics import confusion_matrix, classification_report

predictions = clf.predict(X_test)
print classification_report(y_test, map(lambda x: 1 if x >= 0. else -1, predictions))
print confusion_matrix(y_test, map(lambda x: 1 if x >= 0. else -1, predictions))