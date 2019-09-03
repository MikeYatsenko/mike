import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
df = pd.read_excel('C:\\Users\\Mike\\Desktop\\pi.xlsx', sheet_name='Лист1')
c1 = list(df.iloc[: , 0])
c2 = list(df.iloc[: , 1])
c3 = [c2[i] / c1[i] for i in range(len(c1))]
mean = np.mean(c3)

def stdev(nums):
    diffs = 0
    avg = sum(nums)/len(nums)
    for n in nums:
        diffs += (n - avg)**(2)
    return (diffs/(len(nums)-1))**(0.5)

def getGrabs(nums, mean, std, grabs = []):
    grabs.clear()
    for i in range(len(nums)):
         grabs.append(np.abs((mean - nums[i]) / std))
    return  grabs

def chechGrabs(nums1, nums2):
    s1= nums1.copy()
    s2= nums2.copy()
    div = [nums2[i] / nums1[i] for  i in range((len(nums2)))]
    mean = np.mean(div)
    std = stdev(div)
    grabs = getGrabs(div, mean, std)
    lessThanGrabs = 0;
    indexsForDelete = []
    for i in range(len(nums1)):
        if grabs[i] > 2.87:
            lessThanGrabs +=1
            indexsForDelete.append(i)
    if  lessThanGrabs == 0:
        return  [s1, s2]
    elif lessThanGrabs != 0 :
        for i in range(len(indexsForDelete)):
           nums1[indexsForDelete[i]] = False
           nums2[indexsForDelete[i]] = False
        n1 =  list(filter(lambda x: x != False, nums1))
        n2 =  list(filter(lambda x: x != False, nums2))
        if not n1:
            return [s1, s2]
        else:
            return chechGrabs(n1, n2)


result=chechGrabs(c1,c2)
print(result)
print(len(result[0]))
for i in range(len(result[0])):
    print(result[1][i])
x = (result[0])
y = (result[1])
z = np.polyfit(result[0], result[1], 1)
p = np.poly1d(z)
print(p)
plt.plot(result[0],p(result[0]),"r--")
plt.scatter(x, y)
plt.show()
yhat = p(x)
ybar = (np.sum(y)/len(y))
reg = np.sum((yhat-ybar)**2)
tot = np.sum((y - ybar)**2)
res = reg / tot
print(res)

x= np.array(result[0])
y= np.array(result[1])
popt, pcov = curve_fit(lambda fx,a,b: a+b*np.log(fx), x, y)
x_linspace = np.linspace(min(x), max(x), 100)
power_y = popt[0]+popt[1]*np.log(x_linspace)
plt.scatter(x, y, label='actual data')
plt.plot(x_linspace, power_y, label='log-fit')
plt.legend()
plt.show()
print(popt)
residuals =  y - (popt[0]+popt[1]*np.log(x))
ss_res = np.sum (residuals**2)
ss_tot = np.sum ((y-np.mean(y))**2)
r = 1-(ss_res/ss_tot)
print(r)

def arrWithLog(arr):
    newArr = []
    for i in range(len(arr)):
        newArr.append(np.log(arr[i]))
    return newArr
xWithLog = arrWithLog(x)
yWithLog = arrWithLog(y)
popt, pcov = curve_fit(lambda x, a, b: np.log(a)  + b * x, xWithLog, yWithLog)
print(popt)
x_linspace = np.linspace(min(x), max(x), 100)
power_y = np.log(popt[0])+popt[1]*np.log(x_linspace)
plt.scatter(x, y, label='actual data')
plt.plot(x_linspace, power_y, label='log-fit')
plt.show()

print(popt)
residuals =  y - (np.log(popt[0])+popt[1]*np.log(x))
ss_res = np.sum (residuals**2)
ss_tot = np.sum ((y-np.mean(y))**2)
r = 1-(ss_res/ss_tot)
print(r)







