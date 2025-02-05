import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor




def load_data(file_path):
    return pd.read_csv(file_path,sep='|',on_bad_lines='skip')

def drop_missing(df):
   numeric_columns = df.select_dtypes(include='number').columns.to_list()
   object_columns = df.select_dtypes(include='object').columns.to_list()
   for column in numeric_columns:
      mean_value = df[column].mean()
      df[column] = df[column].fillna(mean_value)  
   for object in object_columns:
      df[object] = df[object].fillna(mean_value)
   df['NumberOfVehiclesInFleet']=df['NumberOfVehiclesInFleet'].fillna('unknown')
   return df
    

def univarate_numerical(data):
    numeric_columns=data.select_dtypes(include=['number']).columns.to_list() 
    fig,axes=plt.subplots(4,3,figsize=(45,45))
    axes=axes.flatten()
    for i,col in enumerate(numeric_columns):
      print(axes[i-1].hist(data[col]))
      print(axes[i-1].set_title(col))
    plt.tight_layout()
    plt.show()
      
      
def univarate_catergorical(data):
    category_columns=data.select_dtypes(include=['object']).columns.to_list()  
    fig,axes=plt.subplots(4,3,figsize=(45,45))
    axes=axes.flatten()
    for i,col in enumerate(category_columns):
      print(axes[i-1].plot(data[col]))
      print(axes[i-1].set_title(col))
    plt.tight_layout()
    plt.show()
   
def boxplot(data):
    numeric_columns=data.select_dtypes(include=['number']).columns.to_list()  
    fig,axes=plt.subplots(4,3,figsize=(45,45))
    axes=axes.flatten()
    for i,col in enumerate(numeric_columns):
      print(axes[i-1].boxplot(data[col]))
      print(axes[i-1].set_title(col))
    plt.tight_layout()
    plt.show()
    

def IQR_outliers(df):
    numeric_columns = df.select_dtypes(include='number').columns.to_list()
    new_df = df.copy()

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        new_df = new_df[(new_df[col] >= lower_limit) & (new_df[col] <= upper_limit)]

    return new_df
  

def model(x_train,y_train,x_test):
  model=LinearRegression()
  model.fit(x_train,y_train)
  predicted_TotalPremium=model.predict(x_test)
  return predicted_TotalPremium

  
  
