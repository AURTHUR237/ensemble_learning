{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>6</th>\n",
       "      <th>148</th>\n",
       "      <th>72</th>\n",
       "      <th>35</th>\n",
       "      <th>0</th>\n",
       "      <th>33.6</th>\n",
       "      <th>0.627</th>\n",
       "      <th>50</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>116</td>\n",
       "      <td>74</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>25.6</td>\n",
       "      <td>0.201</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>762</th>\n",
       "      <td>10</td>\n",
       "      <td>101</td>\n",
       "      <td>76</td>\n",
       "      <td>48</td>\n",
       "      <td>180</td>\n",
       "      <td>32.9</td>\n",
       "      <td>0.171</td>\n",
       "      <td>63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>763</th>\n",
       "      <td>2</td>\n",
       "      <td>122</td>\n",
       "      <td>70</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>36.8</td>\n",
       "      <td>0.340</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>764</th>\n",
       "      <td>5</td>\n",
       "      <td>121</td>\n",
       "      <td>72</td>\n",
       "      <td>23</td>\n",
       "      <td>112</td>\n",
       "      <td>26.2</td>\n",
       "      <td>0.245</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>765</th>\n",
       "      <td>1</td>\n",
       "      <td>126</td>\n",
       "      <td>60</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.1</td>\n",
       "      <td>0.349</td>\n",
       "      <td>47</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>766</th>\n",
       "      <td>1</td>\n",
       "      <td>93</td>\n",
       "      <td>70</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>30.4</td>\n",
       "      <td>0.315</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>767 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      6  148  72  35    0  33.6  0.627  50  1\n",
       "0     1   85  66  29    0  26.6  0.351  31  0\n",
       "1     8  183  64   0    0  23.3  0.672  32  1\n",
       "2     1   89  66  23   94  28.1  0.167  21  0\n",
       "3     0  137  40  35  168  43.1  2.288  33  1\n",
       "4     5  116  74   0    0  25.6  0.201  30  0\n",
       "..   ..  ...  ..  ..  ...   ...    ...  .. ..\n",
       "762  10  101  76  48  180  32.9  0.171  63  0\n",
       "763   2  122  70  27    0  36.8  0.340  27  0\n",
       "764   5  121  72  23  112  26.2  0.245  30  0\n",
       "765   1  126  60   0    0  30.1  0.349  47  1\n",
       "766   1   93  70  31    0  30.4  0.315  23  0\n",
       "\n",
       "[767 rows x 9 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Question 2\n",
    "df = pd.read_csv('pima-indians-diabetes.data.csv')\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(767, 9)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 767 entries, 0 to 766\n",
      "Data columns (total 9 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   6       767 non-null    int64  \n",
      " 1   148     767 non-null    int64  \n",
      " 2   72      767 non-null    int64  \n",
      " 3   35      767 non-null    int64  \n",
      " 4   0       767 non-null    int64  \n",
      " 5   33.6    767 non-null    float64\n",
      " 6   0.627   767 non-null    float64\n",
      " 7   50      767 non-null    int64  \n",
      " 8   1       767 non-null    int64  \n",
      "dtypes: float64(2), int64(7)\n",
      "memory usage: 54.1 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.ensemble import VotingClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.naive_bayes import GaussianNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X=df.iloc[:,:-1]\n",
    "Y=df.iloc[:,-1]\n",
    "X_scaled = scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>6</th>\n",
       "      <th>148</th>\n",
       "      <th>72</th>\n",
       "      <th>35</th>\n",
       "      <th>0</th>\n",
       "      <th>33.6</th>\n",
       "      <th>0.627</th>\n",
       "      <th>50</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>116</td>\n",
       "      <td>74</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>25.6</td>\n",
       "      <td>0.201</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   6  148  72  35    0  33.6  0.627  50\n",
       "0  1   85  66  29    0  26.6  0.351  31\n",
       "1  8  183  64   0    0  23.3  0.672  32\n",
       "2  1   89  66  23   94  28.1  0.167  21\n",
       "3  0  137  40  35  168  43.1  2.288  33\n",
       "4  5  116  74   0    0  25.6  0.201  30"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0\n",
       "1    1\n",
       "2    0\n",
       "3    1\n",
       "4    0\n",
       "Name: 1, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Splitting the data\n",
    "X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AdaBoostClassifier()"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#TRAINING THE ALGORITHMS\n",
    "adaboost_clf = AdaBoostClassifier()\n",
    "adaboost_clf.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.81      0.81      0.81        97\n",
      "           1       0.68      0.68      0.68        57\n",
      "\n",
      "    accuracy                           0.77       154\n",
      "   macro avg       0.75      0.75      0.75       154\n",
      "weighted avg       0.77      0.77      0.77       154\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = adaboost_clf.predict(X_test)\n",
    "report = classification_report(Y_test,y_pred)\n",
    "print(report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "VotingClassifier(estimators=[('dt', DecisionTreeClassifier()),\n",
       "                             ('nb', GaussianNB()), ('svc', SVC())])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "d_clf = DecisionTreeClassifier()\n",
    "svc_clf = SVC()\n",
    "nb_clf = GaussianNB()\n",
    "voting_clf = VotingClassifier( estimators=[('dt', d_clf), ('nb', nb_clf), ('svc', svc_clf)], voting='hard') \n",
    "voting_clf.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.81      0.86      0.83        97\n",
      "           1       0.73      0.65      0.69        57\n",
      "\n",
      "    accuracy                           0.78       154\n",
      "   macro avg       0.77      0.75      0.76       154\n",
      "weighted avg       0.78      0.78      0.78       154\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = voting_clf.predict(X_test)\n",
    "report3 = classification_report(Y_test,y_pred)\n",
    "print(report3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# VOTING CLASSIFIER IS THE BEST AMONGST THE THREE "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
