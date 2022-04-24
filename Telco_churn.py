#SeniorCitizen,Gender,Partner,Dependets,PhoneService,PaperlesBilling Label Encoding
#MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies
##Contract one-hot drop_first True

# PaymentMethod One-hot encoding
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from datetime import date
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
#Adım 1:
df = pd.read_csv("6-feature_engineering/datasets/Telco-Customer-Churn.csv")
df.head()
df.isnull().sum()
df.count()
del_tenure = df.loc[df["tenure"]==0].index
df.drop(del_tenure, axis=0, inplace=True) "Tenure 0 olanları sil nasıl deriz ?
df["TotalCharges"]=df["TotalCharges"].astype(float)
#df["TotalCharges"] = [float(str(i).replace(","," ",".")) for i in df["TotalCharges"]]
df.describe().T
df.info()
#Sayısal değişkenlerde aykırılık gözükmemektedir.
#İnsulinde %75 ve max değer arasında çok fark olduğu için bir baskılama işlemi yapılabilir
#Adım 2:
def grab_col_names(dataframe, cat_th=5, car_th=10):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]
    #Burada yaptığım for döngüsü ile değişkenler içinde col ile gezip eğer "0" ile aynı aynı türden
    ##yani kategorik türünde bir değişkense cat_cols içine at diyoruz.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "0"] #Bu kısımda ise cat_th değerinden değişken sınıf sayısı
    #olarak dataframe'in değişkenlerinin sınıf sayısı küçükse ve "0" tipine eşit değilse yani kategorik değilse
    ##bunları numerik gözüküp kategorik olanların içine at diyoruz.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"] #Burada ise cat_th 'den sınıf sayısının fazla olması ama tip olarak
    #kategorik gözükmesine rağmen kardinal yani ölçülemez olan değişkenleri ifade etmektedir.
    cat_cols = cat_cols + num_but_cat # Burada kategorik değişkenlerimizi tekrar oluşturma sebebimiz
    #numerik gözüküp kategorik olan değişkenleride buraya eklemektir.
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    #Burada ise kategorik olan ama cardinal olmayanları seç ve kategorik değişkenlerimin içine at diyorum

    #num_cols

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "0"]
    #Burada yaptığımız işlemde değişkenlerin içinden tipi kategorik olmayanları numerik değişkenlere atıyoruz
    #Ayrıca
    num_cols = [col for col in num_cols if col not in num_but_cat]
    #numerik ama kategorik olanların haricindekileride buraya numerik oldukları için ekliyoruz.

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
#Adım 3:
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in (num_but_cat,"customerID")]

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
df.info()
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col,check_outlier(df,col))
    #DiabetesPedigreeFunction ve Pregnancies haricinde True dönmemsi gerekir.

#Adım 5:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[((dataframe[variable] < low_limit)), variable] = dataframe[variable].std()
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    #False Verdi

for col in num_cols:
    replace_with_thresholds(df,col)

#Adım 6 : Eksik değerler
df.isnull().values.any()
#False
#Adım 7: cor analizi

corr = df.corr()
#Tenure ve TotalCharges arasında 0.826 korelasyon
##TotalCharges ile MonthlyCharges arasında ise 0.651 korelasyon mevcut
msno.heatmap(corr)
plt.show()

#Görev 2:
#Adım 1:
#Adım 3:
#BİNARY
def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)
#OHE
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

#Rare
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "Churn", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

rare_analyser(df, "Churn", cat_cols not in "customerID")

rare_ıd.head()= df.drop("customerID",axis=1)
df = rare_encoder(rare_ıd, 0.01)

df.head()
df["customerID"]=dff["customerID"]

#Adım 4 :
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

#Adım 5 : Model

y = df["Churn"] #bağımlı değişken
X = df.drop(["Churn","customerID"], axis=1) #bağımsız değişkenler bunların haricindekiler

#Değişkenlerimizi belirliyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)
#Bu değişkenleri train ve test olarak ik farklı sınıfa ayırıyoruz. Trainler ile model kurup test ile bunları
##denetliyor olacağız

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
#Burada önce modelin x_testini tahmin etmesini istiyoruz.
##Daha sonra biz bu tahmin değerleri ile elimizdeki değerlerin karşılaştırılmasını yaparak
### skorumuzu alıyoruz.
accuracy_score(y_pred,y_test)
#Doğruluk skorunu test edip sonucu almak için ilk modelin bulduğunu ikinci olarak

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)

