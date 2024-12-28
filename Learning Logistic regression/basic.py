import pandas as pd
df = pd.read_csv("PATH OF THE CSV ")



## Quality MAppings::

# a mapping dictionary that maps the quality values from 0 to 5
quality_mapping = {
3: 0,
4: 1,
5: 2,
6: 3,
7: 4,
8: 5
}
# you can use the map function of pandas with
# any dictionary to convert the values in a given
# column to values in the dictionary
df.loc[:, "quality"] = df.quality.map(quality_mapping)


## Splitting the Test and train Data into any kind of N Ratios

# use sample with frac=1 to shuffle the dataframe
# we reset the indices since they change after
# shuffling the dataframe
df = df.sample(frac=1).reset_index(drop=True)

## Select the top 1000 rows for training
## for tarinig
df_train = df.head(1000)


# bottom 599 values are selected
# for testing/validation
df_test = df.tail(599)


