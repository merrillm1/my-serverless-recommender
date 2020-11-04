import numpy as np
import pickle
import scipy.sparse
import joblib
from flask import Flask, request
import sys
import pandas as pd
import flasgger
from flasgger import Swagger


user_to_product_interaction = scipy.sparse.load_npz('user_to_product_interaction.npz')

product_to_feature_interaction = scipy.sparse.load_npz('product_to_feature_interaction.npz')                      								
                     								
user_to_index_mapping = np.load("user_to_index_mapping.pkl", allow_pickle=True)

product_to_feature = pd.read_csv('product_to_feature.csv', index_col=0)

user_mapping = [(k, v) for k, v in user_to_index_mapping.items()]

items = np.load('items.npy', allow_pickle=True)                      			
          
# light_fm model         
filename = 'lightfm_model.pkl'

lightfm_model = joblib.load(filename, mmap_mode=None)

def get_key(val, my_dict):
	for key, value in my_dict.items():
		if val == value:
			return key

class recommendation_sampling:
    
	def __init__(self, model, items = items, user_to_product_interaction_matrix = user_to_product_interaction, 
                user2index_map = user_to_index_mapping):
        
		self.user_to_product_interaction_matrix = user_to_product_interaction_matrix
		self.model = model
		self.items = items
		self.user2index_map = user2index_map
    
	def recommendation_for_user(self, user_num, user_features=None):
        
		user = get_key(int(user_num), user_to_index_mapping)
        
        # getting the userindex
        
		userindex = self.user2index_map.get(user, None)
        
		if userindex == None:
			return None
			
		users = [userindex]
        
		# scores from model prediction
		scores = self.model.predict(user_ids = users, item_ids = np.arange(self.user_to_product_interaction_matrix.shape[1]),
                                    user_features=user_features,
                                    item_features = product_to_feature_interaction)

		# top items
        
		top_items = self.items[np.argsort(-scores)]
        
		item_1 = product_to_feature['feature'][product_to_feature['product_id'] == top_items[:3][0]].iloc[0]
		item_2 = product_to_feature['feature'][product_to_feature['product_id'] == top_items[:3][1]].iloc[0]
		item_3 = product_to_feature['feature'][product_to_feature['product_id'] == top_items[:3][2]].iloc[0]
        
		return item_1, item_2, item_3
	
	def known_likes(self, user_num):
		
		user = str(get_key(int(user_num), user_to_index_mapping))
		
		# getting the userindex
        
		userindex = self.user2index_map.get(user, None)
        
		if userindex == None:
			return None
			
		else:
			# products already bought
			known_positives = self.items[self.user_to_product_interaction_matrix.tocsr()[userindex].indices]
			like_1 = product_to_feature['feature'][product_to_feature['product_id'] == known_positives[:3][0]].iloc[0]

		return like_1
                      			
# Initialise a Flask app
app = Flask(__name__)
Swagger(app)

# Create an API endpoint
@app.route('/predict')

def select_user_and_recommend():
	"""Let's predict some items for a selected user. 
	Choose a user from 0 to 91628.
	---
	parameters:
	  - name: user_num
	    in: query
	    type: number
	    required: true
	responses:
		200:
			description: The output values
	
	"""
	
	user_to_feature_interaction = scipy.sparse.load_npz('user_to_feature_interaction.npz')

	# user_id_selection = SelectField('User', choices=user_mapping)
	user_num = request.args.get('user_num')
	
	recom = recommendation_sampling(model = lightfm_model)
	
	likes = recom.known_likes(user_num)
	result = recom.recommendation_for_user(user_num, user_features=user_to_feature_interaction)
	
	return 'For user ID \n' + str(user_num) + ' who likes items in the ' + str(likes) + \
			' category, I recommend items in the following categories ' + str(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
