## <u> Group Project Assignment: CSU44061 Machine Learning </u>
Jamison Engels - 17300599 - engelsjj@tcd.ie

Matthew Flynn - 17327199 - flynnm20@tcd,ie

Jamie Coffey - 17336373 - coffeyja@tcd.ie

### Introduction: 

### <u> Dataset and Features </u>: - Jamison Engels

The U.K. Department for Transportation provides road accident data through their website that can be downloaded in csv format [2]. The first thing we did was use the pandas function ```info``` to display generic information about the dataset.

```
---  ------             --------------   ----- 
 #   Column             Non-Null Count  Dtype 
---  ------             --------------  ----- 
 0   Accident year      10257 non-null  int64 
 1   Accident severity  10257 non-null  object
 2   Region             10257 non-null  object
 3   Ons code           10257 non-null  object
 4   Speed limit        10257 non-null  object
 5   Light condition    10257 non-null  object
 6   Weather condition  10257 non-null  object
 7   Road surface       10257 non-null  object
 8   Accidents          10257 non-null  int64 

```
From this table we can see that there are no null values in our data and that every value besides ```Accident year``` and ```Accidents``` is stored as a string. Using this information we can begin cleaning and formating the data.

#### Cleaning  
Because our data set was for the year 2018, we decided to drop the ```Accident year``` column as the only value present was 2018. Furthermore, we found that there were only 11 ```Ons code``` that corresponded to the 11 ```Region``` values, making the ```Ons code``` column redunant. We also felt that our data set would be easier to work with if each row correspond to one accident rather than having a count for each accident. We used panda functions to break apart our data and increased the size of the set from 10k values to 24k values:

```
df = df.loc[np.repeat(df.index.values, df["Accidents"])]
```
####  Data to Features

Before mapping our data to features, we first wanted to make sure all the values we were using had some sort of significance to avoid wasting time using insignificant data. We decided to use a chi-squared test for each column that we planned on using as a feature. While it would be too lengthy to go through each of the 40 test that we conducted on each class of each column, we generally found that every column was statistically significant to the severity of the crash with a 95% CI. Our code and method for running these tests can be found within the ```def chi_sqr_test()``` function of our code appendix. From this test, we felt that we should try using all 5 columns of data as features and sought to encode them.      

#### Encoding 
After examining the dataset, we felt that that using <i> One-hot Encoding </i> would be the best way to encode most of our textual data. For values such as ```Region``` and ```Weather condition``` where there is no rank between values, we want there to be the same distance between feature vectors. Through the use of <i> One-hot Encoding </i> each values has the same  Euclidean distance of $\sqrt{2}$. The only value we decided to use <i> Ordinal Encoding </i> for was the ```Speed limit``` value as "1-20 mph" should be ranked below "31-40 mph", with "Highway" as the fastest. For the accident severity, the value we are trying to predict, we also felt that a <i> Label Encoder </i> that maintains rank would be best as "Fatal" should be ranked higher than "Slight". To encode our dataset we used a ```ColumnTransformer``` pipeline as passed in the encoding method and the values we wanted to encode for our input and also label encoded our output. 

```
 pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = LabelEncoder().fit_transform(output_data)
```    

The final result was an input array with 34 parameters and an output array with 3 classes. 

### Methods:


### Experiments:  

### Results:

### Dicussion: 

### Summary: 

### Contributions:  