import streamlit as st
import pandas as pd
import numpy as np
import opendatasets as od
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


st.write(""" 
# Penguin species prediction app!

The dataset is obtained from [Kaggle](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data?resource=download&select=penguins_size.csv)

""")

st.write("""---""")


# def download_dataset(path):
#     od.download(path)

# if st.sidebar.button('Download penguin data, only if its unavaillable', type = "primary"):
#     path = "https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data?resource=download&select=penguins_size.csv"
#     download_dataset(path)


df = pd.read_csv("palmer-archipelago-antarctica-penguin-data\penguins_size.csv")

# Display head/ tail of the penguin dataset

# st.markdown('## Display First five rows of uncleaned dataset')
# st.dataframe(df.head(), use_container_width = True)


df2 = df.copy()

# https://stackoverflow.com/questions/40755680/how-to-reset-index-pandas-dataframe-after-dropna-pandas-dataframe
df2.dropna(axis = 0, inplace = True, ignore_index = True)


# st.markdown('## Display First five rows of cleaned dataset')
# st.dataframe(df2.head(), use_container_width = True)

print('\n**************  df2.isnull().any().sum()  ***************\n')
# print('df2.isnull().any().sum(): ', df2.isnull().any().sum())

# print("https://stackoverflow.com/questions/45836794/selecting-string-columns-in-pandas-df-equivalent-to-df-select-dtypes")
# print('https://stackoverflow.com/questions/66580608/meaning-of-sparse-false-pre-processing-data-with-onehotencoder')

df3 = df2.copy()
df3['sex'] = df3['sex'].str.lower()


# https://stackoverflow.com/questions/60284792/how-to-transform-categorical-column-using-one-hot-encoder-in-sklearn
# https://www.ritchieng.com/machinelearning-one-hot-encoding/
cat_col_to_encode = ['island', 'sex']
onehot_encoder = OneHotEncoder(drop = 'first', handle_unknown = 'ignore')
onehot_encoder.fit(df3[cat_col_to_encode])
onehot_encoded = onehot_encoder.transform(df3[cat_col_to_encode]).todense()
onehot_encoded_df = pd.DataFrame(onehot_encoded, columns = onehot_encoder.get_feature_names_out())
df3 = pd.concat([df3, onehot_encoded_df], axis = 1)

# , sparse_output = False
# print('onehot_encoded: \n', df3)
# print('df3.isnull().any().sum(): ', df3.isnull().any().sum())

# Drop the original categorical columns
df3 = df3.drop(cat_col_to_encode, axis = 1)
print(f"Encoded dataframe : \n{df3}")

st.markdown('## Display First five rows of the transformed dataset')
st.dataframe(df3.head(), use_container_width = True)


X = df3.iloc[:, 1:]
y = df3.iloc[:, 0].astype(str)

print('X: \n', X)
print('y: \n', y)

# ordinal encode target variable

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

print('y transformed : \n', y)


# split the dataset into train and test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# define the model
model = RandomForestClassifier()
# fit on the training set
model.fit(X, y)
model_y_pred = model.predict(X)
acc_score = accuracy_score(y, model_y_pred)

# save the model to disk
filename = 'rf_penguin.pkl'
pickle.dump(model, open(filename, 'wb'))


column_names = X.columns
print(column_names)


# st.markdown('## describe the input X')
# st.dataframe(X.describe(), use_container_width = True)


# User input sidebar
st.sidebar.header('Please select feature value')


culmen_length = [X['culmen_length_mm'].min(), X['culmen_length_mm'].max(), 
                X['culmen_length_mm'].mean()]
culmen_depth = [X['culmen_depth_mm'].min(), X['culmen_depth_mm'].max(), 
                X['culmen_depth_mm'].mean()]
flipper_length = [X['flipper_length_mm'].min(), X['flipper_length_mm'].max(), 
                X['flipper_length_mm'].mean()]
body_mass = [X['body_mass_g'].min(), X['body_mass_g'].max(), X['body_mass_g'].mean()]


# this function takes feature value from user
uploaded_file = st.sidebar.file_uploader("please upload your input features as .csv file", type = ['csv'])
if uploaded_file is not None:
    input_x = pd.read_csv(uploaded_file)

def select_input_features():
    island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    culmen_length_mm = st.sidebar.slider('culmen_length', culmen_length[0], culmen_length[1], culmen_length[2]),
    culmen_depth_mm = st.sidebar.slider('culmen_depth', culmen_depth[0], culmen_depth[1], culmen_depth[2]),
    flipper_length_mm = st.sidebar.slider('flipper_length', flipper_length[0], flipper_length[1], flipper_length[2]),
    body_mass_g =st.sidebar.slider('body_mass', body_mass[0], body_mass[1], body_mass[2])

    data = {'culmen_length_mm': culmen_length_mm, 
            'culmen_depth_mm': culmen_depth_mm, 
            'flipper_length_mm': flipper_length_mm, 
            'body_mass_g': body_mass_g, 
            'island': island, 
            'sex': sex}

    selected_df = pd.DataFrame(data, index = [0])
    return selected_df

input_x = select_input_features()

print('input_x: \n', input_x)

st.subheader('Displaying Features selected by user')

# if uploaded_file is not None:
#     st.write(input_x)
# else:
st.write('**Selected raw features by users**')
st.write(input_x)

cat_col_to_encode = ['island', 'sex']

onehot_encoded_input_x = onehot_encoder.transform(input_x[cat_col_to_encode]).todense()
onehot_encoded_input_x_df = pd.DataFrame(onehot_encoded_input_x, columns = onehot_encoder.get_feature_names_out())
input_x = pd.concat([input_x, onehot_encoded_input_x_df], axis = 1)

input_x = input_x.drop(cat_col_to_encode, axis = 1)
print(f"Encoded input given by user : \n{input_x}")

st.write(' ')
st.write('**Transformed features to feed into the ML model**')
st.write(input_x)


st.write(' ')
st.write(' ')
st.write('**This model has an accuracy score of :** ', acc_score*100,'%')


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_hat = loaded_model.predict(input_x)
print('y_hat :', y_hat[0])
st.write('**predicted label :**', y_hat[0])

y_hat_prob = loaded_model.predict_proba(input_x)
# st.write('y_hat_prob : ', y_hat_prob)

penguin_species = df3['species'].unique()
print('y_hat_prob :', y_hat_prob)

pred_To_Class_map = pd.DataFrame(y_hat_prob, columns = le.classes_)

# le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(pred_To_Class_map)
st.write('**Predicted probabilities mapping to class label**')
st.write(pred_To_Class_map)


print('Predicted species is','"', le.inverse_transform(y_hat)[0], '"', 'with ', 
         y_hat_prob[0].max(), 'probability :)')
st.write('**Predicted species is**','"', le.inverse_transform(y_hat)[0], '"', '**with** ', 
         y_hat_prob[0].max(), '**probability  :)**')



# how to get/ generate requirements.txt
# https://stackoverflow.com/questions/57907655/how-to-use-pipreqs-to-create-requirements-txt-file
# pip install pipreqs
# python -m pipreqs.pipreqs




# https://stackoverflow.com/questions/68233466/shap-exception-additivity-check-failed-in-treeexplainer
# Create Tree Explainer object that can calculate shap values

# explainer = shap.TreeExplainer(loaded_model)
# shap_values = explainer.shap_values(X)
# st.write()
# st.write('---')
# st.header('Feature Importance')
# explainer = shap.Explainer(loaded_model, X)
# shap_values = explainer(X, y = y, check_additivity = False)
# st.pyplot(shap.summary_plot(shap_values, feature_names = X.columns, plot_type = "bar"), bbox_inches = 'tight', dpi = 300, pad_inches = 0)
