# Anime_Score_prediction
Machine Learning project, model which helps to predict the score of Anime series.

Dear user!

Here you can see the Machine Learning project devoted to Anime series score prediction. The model is capable to predict the result with a MAE of 0.7579345380059139 (cross-validation score).

I used the dataset ['Anime DataSet 2022'](https://www.kaggle.com/datasets/vishalmane10/anime-dataset-2022). Big thank you to Vishal Mane for this dataset.

How to use:
If you will use this model to assess the score of your anime, please, don't forget to change the path in _train_data_ variable (use to set path to anime-dataset-2022.CSV file). 
Also you need to add in the dataframe the record about the new Anime which you want to assess (in the _X_train_ variable, after string #28). 
Information should include: ['Episodes', 'Release_year', 'Type', 'Studio', 'Release_season', 'Related_Mange', 'Related_anime', 'Tags', 'Content_Warning' , 'staff'(As you can see in string # 98, I use from staff only ': Original Creator',  '': Director'  ': Music' ) ] .Use the original dataset as the reference to insert the right data format. 'Anime DataSet 2022 was scraped from [Anime Planet](https://www.anime-planet.com/) so you can find all desirable information about anime there.

I got the inspiration and want to say a big thank to  'Kono Subarashii Sekai ni Bakuen wo!' and its creators.

Explosion!!)
