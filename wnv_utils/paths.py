# Contains paths to data and submission folders
import os


PROJECT_PATH = "D:/Data_Science_Projects/WNV/"

# data and submission folders
DATA_PATH = os.path.join(PROJECT_PATH, "data")
SUBMISSION_PATH = os.path.join(PROJECT_PATH, "submission")
DATA_TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
DATA_TEST_PATH = os.path.join(DATA_PATH, "test.csv")
DATA_WEATHER_PATH = os.path.join(DATA_PATH, "weather.csv")
DATA_SUBMISSION_PATH = os.path.join(SUBMISSION_PATH, "submission.csv")
