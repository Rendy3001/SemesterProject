
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from transformers import pipeline

#Title and side bar
st.sidebar.header('Dietary restrictions')

def load_data():
    group1 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP1.csv')
    group2 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP2.csv')
    group3 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP3.csv')
    group4 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP4.csv')
    group5 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP5.csv')

    pd.concat([group1, group2, group3, group4, group5], ignore_index=True).to_csv('food_data.csv', index=False)

    food_data = pd.read_csv('food_data.csv')

    return food_data

food_data = load_data()

#Creating selectbar where user can select a food item
#food_item = st.selectbox("Select a food item:", options=food_data['food'].tolist())
#st.write("You selected:", food_item)

#Creating selectbar where user can select dietary resctriction
#restriction = st.multiselect("Select dietary restriction", options= ['Low carbohydrates diet', 'Low cholesterol diet', 'Low sodium diet'])
#st.write("You selected:", restriction)


#Setting thresholds for each dietary resctriction
carbohydrates_threshold = 15
cholesterol_threshold = 0.1
sodium_threshold = 0.15

#Creating columns for every food considering every set restriction
food_data['Low_carbohydrates'] = np.where(food_data['Carbohydrates'] <= carbohydrates_threshold, True, False)
food_data['Low_sodium'] = np.where(food_data['Sodium'] <= sodium_threshold, True, False)
food_data['Low_cholesterol'] = np.where(food_data['Cholesterol'] <= cholesterol_threshold, True, False)

#Importing standard scaler and scaling data to achieve equality in values of different columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Selecting features 
features = ['Caloric Value', 'Carbohydrates', 'Cholesterol', 'Sodium']
data_scaled = scaler.fit_transform(food_data[features]) 

#Combining scaled data with food, low_carbohydrates, low_sodium, low_cholesterol columns
scaled_df = pd.DataFrame(data_scaled, columns=features)
scaled_df = pd.concat([food_data[['food', 'Low_carbohydrates', 'Low_sodium', 'Low_cholesterol']], scaled_df], axis=1)


#Creating pairs based on euclidian distances
euclidean_matrix = euclidean_distances(data_scaled)

recommended_foods_list = []

def recommender_food_restrictions(food, n_recs, restrictions=None):
    #Checking if the food exists in the dataset
    if food in list(scaled_df['food']):
        #Indexing of the selected food
        ix = scaled_df[scaled_df['food'] == food].index[0]

        #Filtering dataset based on restrictions, if provided
        if restrictions:
            # Applying all specified restrictions
            valid_foods = scaled_df
            for restriction in restrictions:
                if restriction in ['Low_sodium', 'Low_cholesterol', 'Low_carbohydrates']:
                    valid_foods = valid_foods[valid_foods[restriction] == True]
        else:
            valid_foods = scaled_df

        #Finding the indices of the valid foods in the original DataFrame
        valid_indices = valid_foods.index.tolist()

        #Computing distances only for valid foods
        distances = [(i, euclidean_matrix[ix, i]) for i in valid_indices if i != ix]
        distances = sorted(distances, key=lambda x: x[1])[:n_recs]

        #Returning the names of the recommended foods
        recommended_foods_list.clear()
        recommended_foods_list.extend([scaled_df.iloc[i]['food'] for i, _ in distances])
        return recommended_foods_list
    else:
        return 'Food not in the dataset'


def main():
    st.title("Food Recommender System")

    #Creating selectbox for selecting a food item
    food_item = st.selectbox("Select a food item:", options=scaled_df['food'].tolist())
    st.write("You selected:", food_item)

    #Creating multiselect for selecting dietary restrictions
    restriction = st.multiselect("Select dietary restriction", options=['Low_carbohydrates', 'Low_cholesterol', 'Low_sodium'])
    st.write("You selected:", restriction)

    #Getting recommendations
    if st.button("Get Recommendations"):
        recommendations = recommender_food_restrictions(food_item, n_recs=3, restrictions=restriction)
        if isinstance(recommendations, list):
            st.write("### Recommended Foods:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write(recommendations)

if __name__ == "__main__":
    main()

#Loading a pre-trained text generation pipeline
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

def generate_recipe(recommended_foods_list):
    prompt = (
        "Write a short recipe using these ingredients: " + ", ".join(recommended_foods_list)
    )

    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]

recipe = generate_recipe(recommended_foods_list)


st.write(recipe)