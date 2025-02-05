




def split(df):
    x=df.drop('TotalPremium',axis=1)
    y=df['TotalPremium']
    x_train,x_test, y_train,y_test =train_test_split(x,y,train_size=0.3,random_state=42)
    return (x_train, y_train),(x_test,y_test)
    
def model(x_train, y_train):
    model=model.fit(x_train, y_train)
    return model

    