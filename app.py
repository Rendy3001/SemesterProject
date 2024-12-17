import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from transformers import pipeline
from scipy.stats import zscore


def load_data():
    group1 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP1.csv')
    group2 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP2.csv')
    group3 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP3.csv')
    group4 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP4.csv')
    group5 = pd.read_csv('https://raw.githubusercontent.com/Rendy3001/SemesterProject/main/FOOD-DATA-GROUP5.csv')

    pd.concat([group1, group2, group3, group4, group5], ignore_index=True).to_csv('food_data.csv', index=False)

    food_data = pd.read_csv('food_data.csv')

    #Dropping irrelevant columns
    food_data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

    numerical_features_filtered = ['Caloric Value', 'Fat', 'Monounsaturated Fats', 'Carbohydrates', 'Protein', 'Vitamin B3', 'Magnesium', 'Potassium', 'Phosphorus', 'Cholesterol', 'Sodium']

    z_scores = food_data[numerical_features_filtered].apply(zscore)
    outliers = (z_scores.abs() > 4).sum()
    
    #Deleting outliers
    outliers_indices = (z_scores.abs() > 4).any(axis=1)
    food_data = food_data[~outliers_indices]

    return food_data

food_data = load_data()

#Setting thresholds for each dietary resctriction
carbohydrates_threshold = 15
cholesterol_threshold = 0.1
sodium_threshold = 0.15

#Creating columns for every food considering every set restriction
food_data['Low_carbohydrates'] = np.where(food_data['Carbohydrates'] <= carbohydrates_threshold, True, False)
food_data['Low_sodium'] = np.where(food_data['Sodium'] <= sodium_threshold, True, False)
food_data['Low_cholesterol'] = np.where(food_data['Cholesterol'] <= cholesterol_threshold, True, False)

#Scaling data to achieve equality in values of different columns
scaler = StandardScaler()

#Selecting features
numerical_features_filtered = ['Caloric Value', 'Fat', 'Monounsaturated Fats', 'Carbohydrates', 'Protein', 'Vitamin B3', 'Magnesium', 'Potassium', 'Phosphorus', 'Cholesterol', 'Sodium']
data_scaled = scaler.fit_transform(food_data[numerical_features_filtered]) 

#Combining scaled data with food, low_carbohydrates, low_sodium, low_cholesterol columns
scaled_df = pd.DataFrame(data_scaled, columns=numerical_features_filtered)

#Reseting indeces before combining to ensure both dataframes have same indices
food_data_reset = food_data.reset_index(drop=True)
scaled_df_reset = scaled_df.reset_index(drop=True)

scaled_df = pd.concat([food_data_reset[['food', 'Low_carbohydrates', 'Low_sodium', 'Low_cholesterol']], scaled_df_reset], axis=1)


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
            #Applying all specified restrictions
            valid_foods = scaled_df
            for restriction in restrictions:
                if restriction in ['Low_sodium', 'Low_cholesterol', 'Low_carbohydrates']:
                    valid_foods = valid_foods[valid_foods[restriction] == True]
        else:
            valid_foods = scaled_df

        #Finding the indices of the valid foods 
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

#Loading the pre-trained text generation model
generator = pipeline("text2text-generation", model="google/flan-t5-large")

if "recipes" not in st.session_state:
    st.session_state["recipes"] = []

def save_recipe_to_session(food, recipe):
    #Saving recipe to session state
    st.session_state["recipes"].append({
        'food': food,
        "recipe": recipe 
    })

def display_history_from_session():
    #Displaying previously generated recipes 
    if st.session_state["recipes"]:
        for record in st.session_state["recipes"]:
            st.sidebar.write(f"The inspirational food for this recipe is: {record['food']}.")
            st.sidebar.write(f"Steps to prepare this recipe: {record['recipe']}")
    else:
        st.sidebar.write("No recipes generated yet.")

def generate_recipe(recommended_foods_list):
    if not recommended_foods_list:
        return "No recommended foods to generate a recipe."

    prompt = (
        "Assume the role of an experienced chef and recipe creator. "
        "Your assignment is to develop a recipe. Provide step-by-step instructions. "
        "This recipe should be crafted using these ingredients: " + ', '.join(map(str, recommended_foods_list))
    )

    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"]

def main():
    st.title("Food Recommenddation System and Recipe Generator")
    st.sidebar.header('History of recipes')
    

    #Creating session state for storing recommendations
    if "recommendations" not in st.session_state:
        st.session_state["recommendations"] = []

    #Creating selectbox for selecting a food item
    food_item = st.selectbox("Select a food item:", options=scaled_df['food'].tolist())
    st.write("You selected:", food_item)

    #Mapping for dietary restrictions
    restriction_display_mapping = {
        'Low_carbohydrates': 'Low Carbohydrate Diet',
        'Low_cholesterol': 'Low Cholesterol Diet',
        'Low_sodium': 'Low Sodium Diet'
    }
    restriction_internal_mapping = {v: k for k, v in restriction_display_mapping.items()}

    #Creating multiselect option for restrictions
    selected_restrictions_display = st.multiselect(
        "Select dietary restrictions:",
        options=list(restriction_display_mapping.values())
    )
    selected_restrictions_internal = [
        restriction_internal_mapping[display_name]
        for display_name in selected_restrictions_display
    ]

    #Getting recommended food items
    if st.button("Get Recommendations"):
        recommendations = recommender_food_restrictions(food_item, n_recs=3, restrictions=selected_restrictions_internal)
        if isinstance(recommendations, list):
            st.session_state["recommendations"] = recommendations  
            st.write("### Recommended Foods:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.session_state["recommendations"] = []
            st.write(recommendations)

    #Generating Recipe
    if st.session_state["recommendations"]:
        if st.button("Generate Recipe"):
            recipe = generate_recipe(st.session_state["recommendations"])
            st.write("### Generated Recipe:")
            st.write(recipe)

            #Saving generated recipe to session state
            save_recipe_to_session(food_item, recipe)

    #Displaying recipe history
    display_history_from_session()

if __name__ == "__main__":
    main()
