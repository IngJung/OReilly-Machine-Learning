#pg 46

# This fetches the data
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("dataset","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close

fetch_housing_data()


# This loads the data
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

# Different methods to get a high-level view of the data
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())

%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()

#pg 51
#creating a test set

###inefficient algorithm
import numpy as np

def split_train_test(data,test_ratio):
    np.random.seed(42) #fixes the seed for the permutation
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = split_train_test(housing,0.2)
print(len(train_set))
print(len(test_set))

### efficient algorith that keeps the same test set even when re-uploading
from zlib import crc32

def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32

def split_train_test_by_id(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds an 'index' column
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"index")


###scikit-learn function
from sklearn.model_selection import train_test_split

train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)

#pg 53
#stratified sampling

###divide median income into different categories
housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0.,1.5,3.0,4.5,6.,np.inf],
                                labels=[1,2,3,4,5])
housing["income_cat"].hist()

###scikit-learn function
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

###test that it worked
strat_test_set["income_cat"].value_counts()/len(strat_test_set)

###remove the income_cat attribute to get data back to its original state
for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

#pg 56
#explore the data

###make a copy of the data to play with it without changing the original set
housing = strat_train_set.copy()

###different visualisations of the data
print(housing.plot(kind="scatter",x="longitude",y="latitude"))
print(housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1))
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
                      s=housing["population"]/100,label="population",figsize=(10,7),
                        c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()

###correlation
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

###scatter matrix
from pandas.plotting import scatter_matrix

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

###zoom in
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)

#pg 63
#data cleansing

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#scikit-learn imputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head())

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

print(housing_cat_1hot.toarray())

print(cat_encoder.categories_)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): #no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self #nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

print(housing.values)
print(housing_extra_attribs)

#pg 70
#Transformation pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy="median")),
                ('attribs_adder',CombinedAttributesAdder()),
                ('std_scaler',StandardScaler()),
                ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared[0])

#pg 72
#select and train a model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)