## <u> Group Project Assignment: CSU44061 Machine Learning </u>
Jamison Engels - 17300599 - engelsjj@tcd.ie

Matthew Flynn - 17327199 - flynnm20@tcd,ie

Jamie Coffey - 17336373 - coffeyja@tcd.ie

### Introduction: 

### <u> Dataset and Features </u> - Jamison Engels:

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

#### Encoding 
After examining the dataset, we felt that that using <i> One Hot Encoding </i> would be the best way to encode our textual data. While it can be argued that an <i> Ordinal Encoder </i> would be better for some values like ```Speed limit``` were some value can be generated from maintaing some distance between data points, we felt that 


### Methods:

### Experiments:  

### Results:

### Dicussion: 

### Summary: 

### Contributions:  