import numpy as np
from math  import log, sqrt
from numpy import linalg
# def my_dot_1 (v,A):
#   C = []
#   for j in range(len(A)):
#     C.append([])
#     for i in range(len(A)):
#       C[j].append(A[j][i]*v[i])
#   return np.array(C)
# def my_dot_2 (v,A):
#   C = []
#   for i in range(len(A)):
#     C.append([])
#     for j in range(len(A)):
#       C[i].append(A[j][i]*v[j])
#   return np.array(C)

#Обучающие выборки
X_1 = np.array([
    [0.7295200749361322, 0.7286337326606256],
    [0.826505923560258, 0.9403158114766519],
    [0.6023139955320228, 0.4890999051968531],
    [0.7924301535257385, 0.9724603339895345],
    [0.3739062132424882, 0.3807945141554343]
    ])
X_2 = np.array([
    [0.036908766218045196, 0.23196814194239498],
    [-0.18516646555268645, -0.16324434635897248],
    [0.19909138385183697, -0.20702094026594725],
    [0.2802286875966722, -0.3525718017732991]
  ])
X_3 = np.array([
    [0.8281937591326639, -0.48370952458304206],
    [-0.7858693510608477, -0.23619862036716505],
    [-0.71001136864972, 0.41404672725112407],
    [-0.4664238951665492, 0.7619952612050831],
    [-0.457969679633352, 0.6615982411339161],
    [0.5983181292607914, -0.8221655651644655],
    [0.8221320333779605, -0.15771200790888465],
    [-0.49537312338291506, -0.7711426700819444],
    [-0.7422364964100625, 0.7182224765639817],
    [-0.6377879808015602, 0.5664880970210512],
    [-0.9306163344230406, 0.40367110070906637],
    [-0.8220373822748085, 0.6909977471317875],
    [0.8486753113201562, 0.63804443731353],
    [-0.6326025764414666, -0.5660631348100686],
    [-0.7988878989878563, -0.7972747092759437],
    [0.8665672650446975, 0.09077183191199585],
    [0.8297187116097032, -0.19949614828904294],
    [-0.7881099342086356, 0.5319196165347493],
    [-0.6965345338796621, 0.034560747319591745],
    [-0.5844695109910769, 0.8287781873912904],
    [0.19909138385183697, -0.20702094026594725],
    [0.2802286875966722, -0.3525718017732991]
  ] )
Mark_1 = []
Mark_2 = []
Mark_3 = []
Old_Mark_1 = []
Old_Mark_2 = []
Old_Mark_3 = []
# Вероятность того, что вектор належить розподілу
teta = 0.9
def is_pos_def(x):# положительная определенность матрицы
    return np.all(np.linalg.eigvals(x) > 0)
E = np.array([[1,0],[0,1]])# Дисперсия
E1 = np.array([[-1,0],[0,-1]])
A = np.linalg.inv(E) 
print ("E = \n",A)
m = np.array([0,0])# мат ожидание
print ("m = ",m)
k = 2*log(teta) + m.dot(A.dot(m)) # константа
print ("k = ",k)
print ("E = \n", E)
print ("E1 = \n", E1)
print("E = ", is_pos_def(E))
print ("E1 = ", is_pos_def(E1))
#v_1 = np.array([1,0]) # собственный вектор
#искривля пространство, получаем кси от х и L от A,m,k
def ksi(x):
  """
  >>> ksi([1,0])
  array([1, 0, 0, 0, 1, 1, 0, 0, 1])
  """
  ksi = []
  for j in range (0,2*len(x)):
    for i in range(0,len(x)):
      #print(i+len(x)*j,len(x)*len(x))
      if i+len(x)*j < len(x)*len(x): ksi.append(x[i]*x[j])#, print (x[i]*x[j])
      else: ksi.append(x[j - len(x)]) #, ksi.append(x[j - len(x)])
  ksi.append(1) 
  #print ("ksi(x) = ", ksi)
  return np.array(ksi)

def L (A,m,k):
  """
  >>> L ([[1,0],[0,1]],[0,0],-0.4462871026284194)
  array([ 1.       ,  0.       ,  0.       ,  1.       ,  0.       ,
          0.       ,  0.       ,  0.       , -0.4462871])
  """
  L = []
  for j in range (0,2*len(m)):
    for i in range(0,len(m)):
      if i+len(m)*j < len(m)*len(m): L.append(A[i][j])
      else: L.append(-2*m[i]*A[i][j - len(m)])#, print (m[i],A[i][j - len(m)])
  L.append(k) 
  #print ("L = ", L)
  return np.array(L)
def Kozinec(L,X,k): # k = 1 or  k = - 1
  for i in range (len(X)):
    x = X[i]
    ksi_ = k* np.array(ksi(x))
    if np.dot(L,ksi_) < 0: 
      Gamma= np.array(max((np.dot(-ksi_,L-ksi_))/(np.dot((L-ksi_),(L-ksi_))),0))
      L = np.array(L*Gamma + (1 - Gamma)*ksi_)
      return Kozinec(L,X,k) 
  return L
def Search_Kozinec(X, L):
  """
  >>> Search_Kozinec([[1., 0.],[0., 1]], [ 1.       ,  0.       ,  0.       ,  1.       ,  0.       ,0.       ,  0.       ,  0.       ,1])
  [1, 1]

  """
  Mark = []
  for i in range (len(X)):
    x = X[i]
    ksi_ = ksi(x) 
    #print ("L * ksi_ = ",L.dot(ksi_) )
    if np.dot(L,ksi_) < 0: Mark.append(0)
    else: Mark.append(1)
  return Mark
def L_to_D (L):
  A = []
  n = int(sqrt((len(L)-1)/2))
  for j in range (n):
    A.append([])
    for i in range (n):
      A[j].append(L[j*n+i])
  A = np.array(A)
  #print("A = \n",A)
  return A
def start(L,A,X_1,X_2,X_3,Old_Mark_1,Old_Mark_2,Old_Mark_3,Mark_1,Mark_2,Mark_3,k):
  L = Kozinec(L, X_1,1)
  L = Kozinec(L,-X_2,-1)
  #print ("\nL = ", L)

  #print ("Lesson 1")
  if k > 0:
    Old_Mark_1 = Mark_1
    Old_Mark_2 = Mark_2
    Old_Mark_3 = Mark_3

  Mark_1 = Search_Kozinec(X_1, L)
  Mark_2 = Search_Kozinec(X_2, L)
  Mark_3 = Search_Kozinec(X_3, L)
  print ("\nMark_1 = ", Mark_1)
  print ("\nMark_2 = ", Mark_2)
  print ("\nMark_3 = ", Mark_3)
  if k > 0 and Old_Mark_1 == Mark_1 and Old_Mark_2 == Mark_2 and Old_Mark_3 == Mark_3: 
    print("STOP cause nothing new")
    return L
  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
  A = L_to_D (L)
  if k > 490: 
    print ("STOP")
    return L
  if is_pos_def(A) == 1:
    for i in range (len(Mark_1)):
      for j in range (len(Mark_2)):
        if Mark_1[i] == 0 or Mark_2[j] == 1 : return  start(L,A,X_1,X_2,X_3,Old_Mark_1,Old_Mark_2,Old_Mark_3,Mark_1,Mark_2,Mark_3, k+1)
  if is_pos_def(A) == 0:
    print ("Матрица не положительно определена")
    print(A)
    Z = []
    w,v = linalg.eig(A)
    #print(w,v)
    mask = np.logical_and(w<0,w<0)
    #print(mask)
    for i in range (len(v)):
      if mask[i] == False: 
        Z.append(v[i])
        print("####")
        print("v[",i,"] = ",v[i])
        #A = my_dot_1 (v[i],A)
        #A = my_dot_2 (v[i],A)
        #A = A.dot(v[i].transpose())
    Z = np.array(Z)
    #Kozinec(L,Z,1)
    #print ("A &", is_pos_def(A))
    #print ("Z =\n",Z)
    #print ("X_1 =\n",X_1)
    X_2 = np.concatenate([X_1,Z],axis = 0)
    #print("X_1=",X_1)
    #print ("X_1 =\n",len(X_1))
    #print ("Z =\n",X_3)
    #E = np.array([[1,0],[0,1]])# Дисперсия
    #A = np.linalg.inv(E) 
    # m = np.array([0,0])# мат ожидание
    # k = 2*log(teta) + m.dot(A.dot(m)) # константа
    start(L,A,X_1,X_2,X_3,Old_Mark_1,Old_Mark_2,Old_Mark_3,Mark_1,Mark_2,Mark_3, k+1)
  else: return L

L = L (A,m,k)
C = L
L = start(L,A,X_1,X_2,X_3,Old_Mark_1,Old_Mark_2,Old_Mark_3,Mark_1,Mark_2,Mark_3,0)
# L = start(L,A,X_1,X_2,X_3,Old_Mark_1,Old_Mark_1,Old_Mark_10)
# L = start(L,A,X_1,X_2,X_3,Old_Mark_1,Old_Mark_1,Old_Mark_10)


if __name__ == "__main__":
  import doctest
  doctest.testmod()
