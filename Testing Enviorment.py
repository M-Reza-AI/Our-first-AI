# use this area to examine any line of code you like!

print("\n"+"WELCOME TO THE TESTING ENVIORMENT!"+"\n"+"-"*35+"\n")

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.dataset_download_file('himanshunakrani/iris-dataset',
                          file_name='iris.csv')

Iris_dataset = pd.read_csv('iris.csv')
