print("__________________________________________Starting of Programme________________________________________________")
import pandas as jack
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, plot_confusion_matrix
data = jack.read_csv('testset.csv')
print("______________________________________________First Ten Data____________________________________________________")
print(data.head(10))
print("______________________________________________Last Ten Data_____________________________________________________")
print(data.tail(10))
# Now copying of Data so that PreProcessing can be done..........
data_copy = data.copy()
print("_______________________________________________Size of Data___________________________________________________")
print("Size of Row and Column:", data.shape)
print("_______________________________________________All Columns of Data_____________________________________________")
print(data.columns)
print("_______________________________________________Dataset Information____________________________________________")
print(data_copy.info())
# Now go For PreProcessing of Data..........
print("______________________________________________PreProcessing of Data___________________________________________")
# Drop the features which is not required for prediction..............
print("_________________________________________________No Dropping Features_________________________________________")
# Now go for Duplicate of the Data..........
print("_________________________________________________Duplicated Data_____________________________________________")
print(" the counts is :", data_copy.duplicated().value_counts())
print(data_copy.duplicated())
dropped_data = data_copy.drop_duplicates()
print("_________________________________________Datasets after Dropping Duplicate Values______________________")
print(dropped_data.head(10))
print("_______________________________________________Size of Data___________________________________________________")
print("Size of Row and Column:", dropped_data.shape)
# Converting Datetime Object into Integer.............
dropped_data["datetime_utc"] = jack.to_datetime(dropped_data["datetime_utc"])
print("________________________________________________Object into Int Format___________________________________")
print(dropped_data.info())
# Now Creating New Column in the Dataset for year|month|day|time..............
dropped_data["year"] = dropped_data["datetime_utc"].dt.year
dropped_data["month"] = dropped_data["datetime_utc"].dt.month
dropped_data["day"] = dropped_data["datetime_utc"].dt.day
dropped_data["time"] = dropped_data["datetime_utc"].dt.time
print("_____________________________________________Dataset with New Column_______________________________________")
print(dropped_data.head(10))
print("_______________________________________________Size of Data___________________________________________________")
print("Size of Row and Column:", dropped_data.shape)
print("_______________________________________________Dataset Information____________________________________________")
print(dropped_data.info())
# Now Convert the String Format Data into Category.................
dropped_data["conds"] = dropped_data["conds"].astype("category")
dropped_data["conds"] = dropped_data["conds"].cat.codes
dropped_data["wdire"] = dropped_data["wdire"].astype("category")
dropped_data["wdire"] = dropped_data["wdire"].cat.codes
print("________________________________________________Object into Int Format___________________________________")
print(dropped_data.head(10))
print("_______________________________________________Dataset Information____________________________________________")
print(dropped_data.info())
# Now go for filling the Missing values of Data..........
print("________________________________________________Null Values in the Data________________________________________")
print(dropped_data.isnull())
print("____________________________________Total Null Values in accordance with the Columns________________")
print(dropped_data.isnull().sum())
print("_____________________________________________________Unique Values from Cond_____________________________")
data_unique = dropped_data["conds"].unique()
print(data_unique)
# Now To Find the Mean for Filling the Values select the Rows on the basis of Label.................
print("______________________________________Rows With Different Labels____________________________________________")
print("_____________________________________Segregated Data via Labels_______________________________________")
# Creating a Empty Dataframe and List...........
Dataframe_list = []
B1 = []
B2 = []
B3 = []
B4 = []
B5 = []
B6 = []
B7 = []
B8 = []
B9 = []
B10 = []
B11 = []
# Now Loop runs for 40 times because we have 40 different Categories in the Label Feature..............


def divide_with_rows():
    for i in range(0,40):
        A = dropped_data.loc[dropped_data["conds"] == data_unique[i]]
        # print("_________________________________Labels:_______________________________", data_unique[i])
        # print(A.head())
        B1.append(A["dewptm"].mean())
        B2.append(A["heatindexm"].mean())
        B3.append(A["hum"].mean())
        B4.append(A["precipm"].mean())
        B5.append(A["pressurem"].mean())
        B6.append(A["wspdm"].mean())
        B7.append(A["tempm"].mean())
        B8.append(A["vism"].mean())
        B9.append(A["wdird"].mean())
        B10.append(A["wgustm"].mean())
        B11.append(A["windchillm"].mean())
        # Now Going to Fill the NA Values...............
        A["dewptm"].fillna(A["dewptm"].mean(),inplace=True)
        A["heatindexm"].fillna(A["heatindexm"].mean(), inplace=True)
        A["hum"].fillna(A["hum"].mean(), inplace=True)
        A["precipm"].fillna(A["precipm"].mean(), inplace=True)
        A["pressurem"].fillna(A["pressurem"].mean().round(), inplace=True)
        A["wspdm"].fillna(A["wspdm"].mean(), inplace=True)
        A["tempm"].fillna(A["tempm"].mean(), inplace=True)
        A["vism"].fillna(A["vism"].mean(), inplace=True)
        A["wdird"].fillna(A["wdird"].mean(), inplace=True)
        A["wgustm"].fillna(A["wgustm"].mean(), inplace=True)
        A["windchillm"].fillna(A["windchillm"].mean(), inplace=True)
        # Now Storing the Row Wise Data into List............
        Dataframe_list.append(A)


# Calling the Function.............
divide_with_rows()
# Now find the Mean for Respective Rows in accordance with the Labels................
print("________________________________________Rows Means With Different Labels______________________________________")
print(B1)
print(B2)
print(B3)
print(B4)
print(B5)
print(B6)
print(B7)
print(B8)
print(B9)
print(B10)
print(B11)
# Now Convert the Dataframe List into Dataframe so further Preprocessing can be done..................
print("_______________________________List to DataFrame Conversion__________________________________________")
# print(Dataframe_list)
D1 = jack.DataFrame(Dataframe_list[0])
D2 = jack.DataFrame(Dataframe_list[1])
D3 = jack.DataFrame(Dataframe_list[2])
D4 = jack.DataFrame(Dataframe_list[3])
D5 = jack.DataFrame(Dataframe_list[4])
D6 = jack.DataFrame(Dataframe_list[5])
D7 = jack.DataFrame(Dataframe_list[6])
D8 = jack.DataFrame(Dataframe_list[7])
D9 = jack.DataFrame(Dataframe_list[8])
D10 = jack.DataFrame(Dataframe_list[9])
D11 = jack.DataFrame(Dataframe_list[10])
D12 = jack.DataFrame(Dataframe_list[11])
D13 = jack.DataFrame(Dataframe_list[12])
D14 = jack.DataFrame(Dataframe_list[13])
D15 = jack.DataFrame(Dataframe_list[14])
D16 = jack.DataFrame(Dataframe_list[15])
D17 = jack.DataFrame(Dataframe_list[16])
D18 = jack.DataFrame(Dataframe_list[17])
D19 = jack.DataFrame(Dataframe_list[18])
D20 = jack.DataFrame(Dataframe_list[19])
D21 = jack.DataFrame(Dataframe_list[20])
D22 = jack.DataFrame(Dataframe_list[21])
D23 = jack.DataFrame(Dataframe_list[22])
D24 = jack.DataFrame(Dataframe_list[23])
D25 = jack.DataFrame(Dataframe_list[24])
D26 = jack.DataFrame(Dataframe_list[25])
D27 = jack.DataFrame(Dataframe_list[26])
D28 = jack.DataFrame(Dataframe_list[27])
D29 = jack.DataFrame(Dataframe_list[28])
D30 = jack.DataFrame(Dataframe_list[29])
D31 = jack.DataFrame(Dataframe_list[30])
D32 = jack.DataFrame(Dataframe_list[31])
D33 = jack.DataFrame(Dataframe_list[32])
D34 = jack.DataFrame(Dataframe_list[33])
D35 = jack.DataFrame(Dataframe_list[34])
D36 = jack.DataFrame(Dataframe_list[35])
D37 = jack.DataFrame(Dataframe_list[36])
D38 = jack.DataFrame(Dataframe_list[37])
D39 = jack.DataFrame(Dataframe_list[38])
D40 = jack.DataFrame(Dataframe_list[39])
# Now append one DataFrame into other Row-Wise..............
print("____________________________________________DataFrame with Values_____________________________________________")
L1 = D1.append(D2, ignore_index=True)
L2 = L1.append(D3, ignore_index=True)
L3 = L2.append(D4, ignore_index=True)
L4 = L3.append(D5, ignore_index=True)
L5 = L4.append(D6, ignore_index=True)
L6 = L5.append(D7, ignore_index=True)
L7 = L6.append(D8, ignore_index=True)
L8 = L7.append(D9, ignore_index=True)
L9 = L8.append(D10, ignore_index=True)
L10 = L9.append(D11, ignore_index=True)
L11 = L10.append(D12, ignore_index=True)
L12 = L11.append(D13, ignore_index=True)
L13 = L12.append(D14, ignore_index=True)
L14 = L13.append(D15, ignore_index=True)
L15 = L14.append(D16, ignore_index=True)
L16 = L15.append(D17, ignore_index=True)
L17 = L16.append(D18, ignore_index=True)
L18 = L17.append(D19, ignore_index=True)
L19 = L18.append(D20, ignore_index=True)
L20 = L19.append(D21, ignore_index=True)
L21 = L20.append(D22, ignore_index=True)
L22 = L21.append(D23, ignore_index=True)
L23 = L22.append(D24, ignore_index=True)
L24 = L23.append(D25, ignore_index=True)
L25 = L24.append(D26, ignore_index=True)
L26 = L25.append(D27, ignore_index=True)
L27 = L26.append(D28, ignore_index=True)
L28 = L27.append(D29, ignore_index=True)
L29 = L28.append(D30, ignore_index=True)
L30 = L29.append(D31, ignore_index=True)
L31 = L30.append(D32, ignore_index=True)
L32 = L31.append(D33, ignore_index=True)
L33 = L32.append(D34, ignore_index=True)
L34 = L33.append(D35, ignore_index=True)
L35 = L34.append(D36, ignore_index=True)
L36 = L35.append(D37, ignore_index=True)
L37 = L36.append(D38, ignore_index=True)
L38 = L37.append(D39, ignore_index=True)
L39 = L38.append(D40, ignore_index=True)
# Now its time to make new DataFrame....................
Data_new = L39.copy()
print("______________________________________________First Ten Data____________________________________________________")
print(Data_new.head(10))
print("______________________________________________Last Ten Data_____________________________________________________")
print(Data_new.tail(10))
print("_______________________________________________Size of Data___________________________________________________")
print("Size of Row and Column:", Data_new.shape)
print("_______________________________________________Dataset Information____________________________________________")
print(Data_new.info())
print("____________________________________Total Null Values in accordance with the Columns________________")
print(Data_new.isnull().sum())
# Now Replace the NaN value with their corresponding means................
Data_new["heatindexm"].fillna(Data_new["heatindexm"].mean(), inplace=True)
Data_new["hum"].fillna(Data_new["hum"].mean(), inplace=True)
Data_new["precipm"].fillna(Data_new["precipm"].mean(), inplace=True)
Data_new["tempm"].fillna(Data_new["tempm"].mean(), inplace=True)
Data_new["vism"].fillna(Data_new["vism"].mean(), inplace=True)
Data_new["wgustm"].fillna(Data_new["wgustm"].mean(), inplace=True)
Data_new["windchillm"].fillna(Data_new["windchillm"].mean(), inplace=True)
print("_______________________________________________Good Dataset_____________________________________________")
print(Data_new.head(15))
print("____________________________________Total Null Values in accordance with the Columns________________")
print(Data_new.isnull().sum())
# Now Segregating the Data on the basis of Months.............
print("_____________________________________________________Unique Values from Cond_____________________________")
data_unique = Data_new["month"].unique()
print(data_unique)
# Now go for Visualizing the Data to get the best Experience from the Dataset.............
# First go for Conds Distribution over a different parameters.......................
print("________________________________Conds Distribution______________________________________________")
Data_new.to_csv('testset1.csv')
print("________________Plotting Year and Month with respect to Conds__________________________________________")
Data_new = jack.read_csv('testset2.csv')
print(Data_new.head(10))
Data_new = Data_new.drop(columns = ["datetime_utc", "time"])
print(Data_new)
data_unique = Data_new["month"].unique()
print(data_unique)
data_unique_1 = Data_new["year"].unique()
print(data_unique_1)
print("_______________________________________On the Basis of Month Divide the Dataset_______________________________")
month_1 = Data_new.loc[Data_new["month"] == 1]
month_2 = Data_new.loc[Data_new["month"] == 2]
month_3 = Data_new.loc[Data_new["month"] == 3]
month_4 = Data_new.loc[Data_new["month"] == 4]
month_5 = Data_new.loc[Data_new["month"] == 5]
month_6 = Data_new.loc[Data_new["month"] == 6]
month_7 = Data_new.loc[Data_new["month"] == 7]
month_8 = Data_new.loc[Data_new["month"] == 8]
month_9 = Data_new.loc[Data_new["month"] == 9]
month_10 = Data_new.loc[Data_new["month"] == 10]
month_11 = Data_new.loc[Data_new["month"] == 11]
month_12 = Data_new.loc[Data_new["month"] == 12]
print("_______________________________________On the Basis of Year Divide the Dataset_______________________________")
year_1 = Data_new.loc[Data_new["year"] == 1996]
year_2 = Data_new.loc[Data_new["year"] == 1997]
year_3 = Data_new.loc[Data_new["year"] == 1998]
year_4 = Data_new.loc[Data_new["year"] == 1999]
year_5 = Data_new.loc[Data_new["year"] == 2000]
year_6 = Data_new.loc[Data_new["year"] == 2001]
year_7 = Data_new.loc[Data_new["year"] == 2002]
year_8 = Data_new.loc[Data_new["year"] == 2003]
year_9 = Data_new.loc[Data_new["year"] == 2004]
year_10 = Data_new.loc[Data_new["year"] == 2005]
year_11 = Data_new.loc[Data_new["year"] == 2006]
year_12 = Data_new.loc[Data_new["year"] == 2007]
year_13 = Data_new.loc[Data_new["year"] == 2008]
year_14 = Data_new.loc[Data_new["year"] == 2009]
year_15 = Data_new.loc[Data_new["year"] == 2010]
year_16 = Data_new.loc[Data_new["year"] == 2011]
year_17 = Data_new.loc[Data_new["year"] == 2012]
year_18 = Data_new.loc[Data_new["year"] == 2013]
year_19 = Data_new.loc[Data_new["year"] == 2014]
year_20 = Data_new.loc[Data_new["year"] == 2015]
year_21 = Data_new.loc[Data_new["year"] == 2016]
year_22 = Data_new.loc[Data_new["year"] == 2017]
year_23 = Data_new.loc[Data_new["year"] == 2018]
year_24 = Data_new.loc[Data_new["year"] == 2019]
year_25 = Data_new.loc[Data_new["year"] == 2020]
# Now go for Visualizing the Data to get the best Experience from the Dataset.............
# First go for Conds Distribution over a different parameters.......................
print("________________________________Conds Distribution______________________________________________")
print("__________________________Plotting Year and Month with respect to Conds________________________________")
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(Data_new["year"],Data_new["conds"],c='blue')
axs[0, 1].bar(Data_new["year"], Data_new["conds"], width = 0.8, color = ['red', 'green', 'yellow', 'blue'])
axs[1, 0].scatter(Data_new["month"], Data_new["conds"], c='red')
axs[1, 1].bar(Data_new["month"], Data_new["conds"], width = 0.8, color = ['red', 'green', 'yellow', 'blue'])
axs[0, 0].set_title("Labels vs Year")
axs[0, 1].set_title("Labels vs Year")
axs[1, 0].set_title("Labels vs Month")
axs[1, 1].set_title("Labels vs Month")
plt.show()
ax = month_1.plot(kind="scatter", x="dewptm", y="conds", color="blue", label="Labels vs. Dew")
month_1.plot(kind="scatter", x="hum", y="conds", color="red", label="Labels vs. Hum", ax=ax)
month_1.plot(kind="scatter", x="pressurem", y="conds", color="green", label="Labels vs. Pressure", ax=ax)
month_1.plot(kind="scatter", x="heatindexm", y="conds", color="orange", label="Labels vs. Heatindex", ax=ax)
month_1.plot(kind="scatter", x="rain", y="conds", color="purple", label="Labels vs. Rain", ax=ax)
ax.set_xlabel("horizontal label")
ax.set_ylabel("vertical label")
plt.show()
ax = year_1.plot(kind="scatter", x="dewptm", y="conds", color="blue", label="Labels vs. Dew")
year_1.plot(kind="scatter", x="hum", y="conds", color="red", label="Labels vs. Hum", ax=ax)
year_1.plot(kind="scatter", x="pressurem", y="conds", color="green", label="Labels vs. Pressure", ax=ax)
year_1.plot(kind="scatter", x="heatindexm", y="conds", color="orange", label="Labels vs. Heatindex", ax=ax)
year_1.plot(kind="scatter", x="rain", y="conds", color="purple", label="Labels vs. Rain", ax=ax)
ax.set_xlabel("horizontal label")
ax.set_ylabel("vertical label")
plt.show()
fig, axs = plt.subplots(3, 3)
axs[0, 0].scatter(year_22["tempm"],year_22["conds"], c='blue')
axs[0, 1].scatter(year_21["tempm"], year_21["conds"], c='red')
axs[0, 2].scatter(year_20["tempm"], year_20["conds"], c='yellow')
axs[1, 0].scatter(year_19["tempm"], year_19["conds"], c='green')
axs[1, 1].scatter(year_18["tempm"], year_18["conds"], c='orange')
axs[1, 2].scatter(month_1["tempm"], month_1["conds"], c='blue')
axs[2, 0].scatter(month_2["tempm"], month_2["conds"], c='yellow')
axs[2, 1].scatter(month_3["tempm"], month_3["conds"], c='green')
axs[2, 2].scatter(month_4["tempm"], month_4["conds"], c='orange')
axs[0, 0].set_title("Labels (2017) Temp")
axs[0, 1].set_title("Labels (2016) Temp")
axs[0, 2].set_title("Labels (2015) Temp")
axs[1, 0].set_title("Labels (2014) Temp")
axs[1, 1].set_title("Labels (2013) Temp")
axs[1, 2].set_title("Labels (Jan.) Temp")
axs[2, 0].set_title("Labels (Feb.) Temp")
axs[2, 1].set_title("Labels (March) Temp")
axs[2, 2].set_title("Labels (April) Temp")
plt.show()
# setting the parameter values
annot = True
hm = sn.heatmap(data=year_1, annot=annot)
plt.show()
sn.heatmap(data=month_1, annot=True)
plt.show()
# First go for Conds Distribution over a different parameters.......................
print("________________________________Conds Distribution______________________________________________")
print("________________Plotting Year and Month with respect to Conds__________________________________________")
# Now go for divide the Dependent and Independent variable from the Datasets..................
X = Data_new.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20, 21]].values
y = Data_new.iloc[:, 0].values
print("___________________________________________________Independent Variable________________________________________")
# print(X)
print("______________________________________________Dependent Variable_____________________________________________")
# print(y)
# Now go for print the divided Datasets into Training and Testing Datasets........................
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("_____________________________________________Input Training Datasets_______________________________________")
print(X_train)
print("________________________________________Input Testing Datasets____________________________________________")
print(X_test)
print("___________________________________________Output Training Datasets_________________________________________")
print(y_train)
print("______________________________________________Output Testing Datasets_________________________________________")
print(y_test)
# Now go for Feature Scaling of the Datasets......................
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Now go for Training the Decision Tree Random Forest Algorithm assuming 20 decision trees initially...................
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
print("_________________________**********************************************************____________________________")
print("_________________________**********************************************************____________________________")
print("_______________________________________Predicted Output for Testing Datasets________________________________")
y_pred = dtr.predict(X_test)
print(y_pred)
# Now go for Prediction using user Input...............
pred_input = jack.read_csv("testset3.csv")
print("________________________________________________User Input____________________________________________________")
print(pred_input)
pred_output = dtr.predict(pred_input)
pred_input["Weather_Report"] = pred_output
pred_input.to_csv("Weather_Report.csv")
# Now go for Printing  Weather_Report..................
pred_input_1 = jack.read_csv("Weather_Report.csv")
print("________________________________________________Weather_Report____________________________________________________")
print(pred_input_1)
# Now go for ML Model for Weather Prediction Efficiency.............................
print("___________________________________________R2_Score______________________________________________")
print(r2_score(y_test, y_pred))
print("_______________________________________OVERALL CLASSIFICATION REPORT______________________________________")
print(classification_report(y_test.round(), y_pred.round()))
print("_________________________________________ACCURACY____________________________________________________")
print(accuracy_score(y_test.round(), y_pred.round()))
print("_________________________**********************************************************____________________________")
print("_________________________**********************************************************____________________________")































