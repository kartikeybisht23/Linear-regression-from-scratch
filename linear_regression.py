np.random.seed(0)
import numpy as np
import matplotlib.pyplot as plt
# creating random marks of english maths and science
X=np.random.randint(10,35,(200,3))
#creatinf random grade
Y=np.random.randint(5,10,(200,1))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=1)
#making the regressor
from sklearn import linear_model
rg=linear_model.LinearRegression()
rg.fit(x_train,y_train)
#score
print(f"train score is {rg.score(x_train,y_train)}")
print(f"test score is {rg.score(x_test,y_test)}")
#printing coefficient
print(f"the coeff are{rg.coef_}")
print(rg.predict(x_test))
plt.scatter(y_test,rg.predict(x_test),color="r")
plt.xlabel("y_test")
plt.ylabel("y_pred")

#self algo
def estimate_coeff(x,y):
    #creating b slope values:
    b=np.zeros(x.shape[1])
    
    for i in range (x.shape[1]):
        n=np.size(x[:,i])
        # mean
        x_=np.mean(x[:,i])
        y_=np.mean(y)
        #calculating cross deviation
        sxy=np.sum(x*y)-n*(x_*y_)
        sxx=np.sum(x*x)-n*(x_*x_)
        b[i]=sxy/sxx
        b0=np.mean(y)-(b[0]*np.mean(x[:,0])+b[1]*np.mean(x[:,1])+b[2]*np.mean(x[:,2]))
    return b,b0
def plotting_regression(x,y,b,b0):
    #predicting values
    y_pred=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y_pred[i]=b0+b[0]*x[i,0]+b[1]*x[i,1]+b[2]*x[i,2]
    print(x_train.shape,y_train.shape)
    plt.scatter(y,y_pred, color = "b",marker = "o", s = 30)
    # putting labels 
    plt.xlabel('y_test') 
    plt.ylabel('y_predct')
  
    # function to show plot 
    plt.show()
    return y_pred
def accuracy(y,y_pred):
    r_squarred=1-((np.sum((y-y_pred)**2))/(np.sum((y-(np.mean(y)))**2)))
    print(f"accuracy is {r_squarred}")
def main():
    
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(0)
    # creating random marks of english maths and science
    X=np.random.randint(10,35,(200,3))
    #creatinf random grade
    Y=np.random.randint(5,10,(200,1))
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=1)
    b,b0=estimate_coeff(x_train,y_train)
    print(f"slope are {b}")
    print(f"coeff y intercpt is {b0}")
    y_pred=plotting_regression(x_test,y_test,b,b0)
    accuracy(y_test,y_pred)
    
if __name__=="__main__":
    main()
    
