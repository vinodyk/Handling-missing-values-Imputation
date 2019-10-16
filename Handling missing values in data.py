#Check missing values with histogram for NaN, emptystring, ?,-1,-999..
import seaborn as sns
plt.figure(figsize=(15,8))
sns.distplot(datafram.column_name,bins=30)
#load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
#concatenate train and test
train_objs_num = len(train)
y= train['Survived']
dataset = pd.concat(objs=[train.drop(columns=['Survived']),test], axis=0)
dataset.info()
#checking the percentage of missing data by feature
total = dataset.isnull.sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1, kerys=['Ttoal','Percent'])
f, ax = plt.subplots(figsize=(15,6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index,y=missing_data['Percent'])
plt.xlabel('Features',fontsize=15)
plt.ylabel("Percent of missing values", fontsize=15)
plt.title('Percent missing data by feature', fontsize = 15)
missing_data.head()

# method of dropping rows with missing values
dataframe.dropna(inplace=True)
# method to drop only if all values are missing
dataframe.dopna(how='all',inplace=True)
#method to drop only a column
dataframe.dropna(axis=1, inplace=True)

#keep only rows with at least 4 non-na values
dataframe.dropna(thresh=4, inplace=True)

#back fill
dataframe.fillna(method='bfill',inplace=True)
#forward fill
dataframe.fillna(method='ffill',inplace=True)
#method to replace with a constant
dataframe.Column_Name.fillna(-99,inplace=True)
#randomly fill with values close to mean but within one starndard deviation
Column_Name_avg =datafarame['Column_Name'].mean()
Column_Name_std = datafarame['Column_Name'].std()
Column_Name_null_random_list = np.random.randint(Column_Name_avg - Column_Name_std, Column_Name_avg+Column_Name_std, size=Column_Name_null_count)
datafame['Column_Name'][np.isnan(dataframe['Column_Name'])]=Column_Name_null_random_list
datafram['Column_Name'] = datafram['Column_Name'].astype(int)

#for continuous data replace with Mean ,
dataframe.Column_Name.fillna(datafaram.Column_Name.mean(),inplace=True)
#Median
dataframe.Column_Name.fillna(datafaram.Column_Name.median(),inplace=True)
#Most common of Mode values
dataframe.Column_Name.fillna(datafaram.Column_Name.mode()[0],inplace=True)

#Isnull feature
dataframe['is_null_Column_Name']= dataframe.Column_Name.isnull().astype(int)

#simple Mean Median method
train['Age'].fillna(train.groupby('Sex')['Age'].transform("mean"), inplace=True)
#simple Median method
rain['Age'].fillna(train.groupby('Sex')['Age'].transform("median"), inplace=True)

#Pearson correlation of features
colormap = plt.cm.RdBu
plt.figure(figsize=(32,10))
plt.title('Pearson correlation of features',y=1.05, size=15)
sns.heatmap(train.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


#KNN method to replace missing values of emp_length column by finding 3 nearest
from fancyimpute import KNN
# we use dataframe, fancyimpute removes column names
train_cols=list(train)

# use 5 nearest rows to fill missing features
train = pd.DataFrame(KNN(k=5).complete(train))
train.columns = train_cols

#MICE method uses Bayesian ridge regression avoids baises
from fancyimpute import IterativeImputer as MICE
#use MICE to fill missing rows
train_cols=list(loans)
train=pd.DataFrame(MICE(verbose=False).fit_transform(train))
train.columns =train_cols




#Linear regression method
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
data = train[['Pclass','SibSp','Parch','Fare','Age']]
#Step 1: split the dataset that contains the missing values and no missing values are thest and train respectively.
x_train = data[data['Age'].notnull()].drop(columns='Age')
y_train =data[data['Age'].notnull()]['Age']
x_test = data[data['Age'].isnull].drop(columns='Age')
y_test = data[data['Age'].isnull()]['Age']

#step-2 : Train the algorithm
linreg.fit(x_train,y_train)

#step_3: Predict the missing values in the attribute of the test data
predicted = linreg.predict(x_test)

#step-4: Lets obtain the complete dataset by comining with the target attribute.

train.Age[train.Age.isnull()]= predicted
train.info()
