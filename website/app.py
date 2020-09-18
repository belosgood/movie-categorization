from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'film_plots.sqlite')

# be sure model updates using sqlite database with restart.
def update_model(db_path, model, batch_size=10000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from film_plot_db')

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)

        classes = np.array([0, 1])
        X_train = vect.transform(X)
        clf.partial_fit(X_train, y, classes=classes)
        #clf.fit(X, y)
        results = c.fetchmany(batch_size)

    conn.close()
    return None

update_model(db_path=db, model=clf, batch_size=10000)

# Update classifier file
pickle.dump(clf, open(os.path.join(cur_dir,
             'pkl_objects', 'classifier.pkl'), 'wb')
            , protocol=4)

def classify(document):
    label = {0: 'adult-film', 1: 'romantic-comedy'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO film_plot_db (plot, genre, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

class PlotForm(Form):
    movieplot = TextAreaField('',[validators.DataRequired(), validators.length(min=15)])

@app.route('/')
def index():
    form = PlotForm(request.form)
    return render_template('plotform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = PlotForm(request.form)
    if request.method == 'POST' and form.validate():
        plot = request.form['movieplot']
        y, proba = classify(plot)
        return render_template('results.html',
                                content=plot,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('plotform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    plot = request.form['plot']
    prediction = request.form['prediction']

    inv_label = {'adult-film': 0, 'romantic-comedy': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(plot, y)
    sqlite_entry(db, plot, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)
