import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load sample data (replace this with your actual dataset)
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'product_id': [101, 102, 103, 101, 104, 105, 102, 103, 105],
    'rating': [5, 3, 4, 2, 5, 3, 4, 2, 5]
}
df = pd.DataFrame(data)

# Prepare the data for surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Initialize the SVD algorithm (Singular Value Decomposition)
model = SVD()

# Train the model on the training data
model.fit(trainset)

# Make predictions on the test data
predictions = model.test(testset)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# Example: Predict a rating for a specific user and product
user_id = 1
product_id = 104
predicted_rating = model.predict(user_id, product_id).est
print(f'Predicted rating for user {user_id} on product {product_id}: {predicted_rating}')

# Generate top-N recommendations for a specific user
def get_top_n_recommendations(predictions, n=5):
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n_recommendations = get_top_n_recommendations(predictions, n=3)
print("Top-3 recommendations for each user:")
for uid, user_ratings in top_n_recommendations.items():
    print(f'User {uid}: {[iid for (iid, _) in user_ratings]}')
