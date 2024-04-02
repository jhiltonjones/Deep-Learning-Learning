
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from matplotlib.colors import Normalize
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import joblib

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)
        
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


data = pd.read_csv('/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/WildFires_DataSet.csv')

label_encoder = LabelEncoder()
data['CLASS'] = label_encoder.fit_transform(data['CLASS'])

X = data[['NDVI', 'LST', 'BURNED_AREA']].values
y = data['CLASS'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

best_clf = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
model_path = '/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/saved_model.joblib'

joblib.dump(best_clf, model_path)

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), ["{:.2e}".format(g) for g in gamma_range], rotation=45)
plt.yticks(np.arange(len(C_range)), ["{:.2e}".format(c) for c in C_range])
plt.title('Validation accuracy')


plt.savefig('/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/heatmap.png')

plt.close()
