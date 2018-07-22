import pandas as pd
import pickle
import math
import numpy as np
from flask import *
from fuzzywuzzy import process, fuzz
from werkzeug import secure_filename
from sklearn.preprocessing import Normalizer


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def pre_processing():
    def cleaning_data(df_new, df_old, uc):

        df_new = df_new.drop(uc, axis=1)
        df_new = df_new.set_index([df_old.columns[27]])
        df_new = df_new.dropna()
        df_new = df_new.sort_index()

        return pd.DataFrame(df_new)

    def most_similar(data, m):

        most = {m: []}

        for i in range(len(data)):
            a = process.extract(data[i], data, scorer=fuzz.partial_ratio, limit=2)
            b = math.fsum([item[1] for item in a[1:] if type(item[1]) == int])
            most[m].append(b)

        return most[m]

    def avg_similar(data, avr):

        avg = {avr: []}

        for i in range(len(data)):
            a = process.extract(data[i], data, scorer=fuzz.partial_ratio, limit=4)
            b = math.fsum([item[1] for item in a[1:] if type(item[1]) == int])
            avg[avr].append(b / 3)

        return avg[avr]

    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        old = pd.read_excel(f)
        conv = {old.columns[4]: str, old.columns[5]: str, old.columns[8]: str, old.columns[27]: pd.to_datetime}
        new = pd.read_excel(f, converters=conv)

        user_profile = []
        used_col = []
        unused_col = []

        for col in range(len(new.columns)):
            if col == 4 or col == 5 or col == 8 or col == 27:
                user_profile.append(new.columns[col])
            elif col == 31:
                used_col.append(new.columns[col])
            else:
                unused_col.append(new.columns[col])

        new = cleaning_data(new, old, unused_col)
        old = cleaning_data(old, old, unused_col)

        list_date = {'June': list(new[new.index.month == 6].index.unique().format())}

        iterator = {'June': {'Adr_Reg': [], 'Shp_Reg': [], 'Ord_Mail': []}}
        keys = ['Adr_Reg', 'Shp_Reg', 'Ord_Mail']

        for k, us_p, in zip(keys, user_profile):
            for d in list_date['June']:
                iterator['June'][k].append(new.loc[d, us_p])

        most_key = ['Adr_Reg_Most', 'Shp_Reg_Most', 'Ord_Mail_Most']
        avg_key = ['Adr_Reg_Avg', 'Shp_Reg_Avg', 'Ord_Mail_Avg']
        dict_tmp = {'Adr_Reg_Most': [], 'Shp_Reg_Most': [], 'Ord_Mail_Most': [],
                    'Adr_Reg_Avg': [], 'Shp_Reg_Avg': [], 'Ord_Mail_Avg': []}

        for key, most in zip(keys, most_key):
            for series in iterator['June'][key]:
                dict_tmp[most].extend(most_similar(series, most))

        for key, avg in zip(keys, avg_key):
            for series in iterator['June'][key]:
                dict_tmp[avg].extend(avg_similar(series, avg))

        new = pd.DataFrame(dict_tmp)
        new['Result'] = old['Result'].values

        scaler = Normalizer()

        X = scaler.fit_transform(new.drop('Result', axis=1))

        # Load classifier
        filename = 'model/finalized_model.sav'
        clf = pickle.load(open(filename, 'rb'))
        proba = np.array([])
        rec = np.array([])

        for prob in clf.predict_proba(X):
            score = prob[0] / prob[1]
            proba = np.append(proba, score)
            if score >= 2.0:
                rec = np.append(rec, 'FRAUD')
            elif (score > 1.0) and (score < 2.0):
                rec = np.append(rec, 'WEAK FRAUD')
            else:
                rec = np.append(rec, 'NO FRAUD')

        proba = pd.DataFrame(proba, columns=['Prob_Score'])
        rec = pd.DataFrame(rec, columns=['Rec'])

        columns = new.columns
        new = pd.DataFrame(new, columns=columns)
        new = pd.concat([new, proba, rec], axis=1)

        return new.to_html()


if __name__ == '__main__':
    app.run(debug=True)

