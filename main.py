from flask import Flask, render_template, request
import requests
import pandas as pd
import nltk
import json 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)

def query_process(query, recipe_index):
    query_words = str(query).split()
    doc_ids = []
    lemmatizer=WordNetLemmatizer()
    for w in query_words:
        w = lemmatizer.lemmatize(w.lower())
        if w in recipe_index:
            doc_ids.append(recipe_index[w])

    doc_ids  = set([item for sublist in doc_ids for item in sublist])
    return doc_ids
  

def  find_top_similar_indices(query):
    f = open('posting_list.json')
    recipe_index=json.load(f)
    # user input
    doc_ids = list(query_process(query, recipe_index))

    lemmatizer=WordNetLemmatizer()
    corpus = []

    df = pd.read_json('dataframe.json')

    # print(df.head())
    for d in doc_ids:
        corpus.append(' '.join(df['new_recipe'][d]))

    query = ' '.join([lemmatizer.lemmatize(w.lower()) for w in query.split()])
    corpus.insert(0,query)

    tf_idf_vect = TfidfVectorizer()
    tf_idf = tf_idf_vect.fit_transform(corpus)
    terms = tf_idf_vect.get_feature_names() 

    cosine_sim = cosine_similarity(tf_idf[0:1], tf_idf).flatten()
    related_docs_indices = cosine_sim.argsort()[:-7:-1]   #top 5

    return_list=[]
    for ind in related_docs_indices[1:]:
        df_ind=doc_ids[ind-1]
        # print(ind, "\n", df.iloc[df_ind])
        return_list.append(df.iloc[df_ind])
    return return_list

@app.route('/', methods =["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form.get("query")
        top_list=find_top_similar_indices(query)

        top_df=pd.DataFrame(top_list)
        dish1, dish2, dish3, dish4, dish5=None, None, None, None, None
        url1, url2, url3, url4, url5=None, None, None, None, None
        if len(top_list)>=1:
            dish1=top_df["name"].iloc[0]
            url1=top_df["url"].iloc[0]

        if len(top_list)>=2:
            dish2=top_df["name"].iloc[1]
            url2=top_df["url"].iloc[1]
        
        if len(top_list)>=3:
            dish3=top_df["name"].iloc[2]
            url3=top_df["url"].iloc[2]

        if len(top_list)>=4:
            dish4=top_df["name"].iloc[3]
            url4=top_df["url"].iloc[3]
            
        if len(top_list)>=5:
            dish5=top_df["name"].iloc[4]
            url5=top_df["url"].iloc[4]
            
        return render_template('similar_dishes.html', top_dishes=top_df.to_json(), dish1=dish1,
        dish2=dish2, dish3=dish3, dish4=dish4, dish5=dish5, url1=url1, url2=url2, url3=url3, url4=url4, url5=url5)
    return render_template('form.html')
    

if __name__ == '__main__':
   app.run(debug=True)