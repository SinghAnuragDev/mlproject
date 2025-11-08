import os
import sys
from dataclasses import dataclass

# --- Import all the models ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            # Split data into X and y
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All rows, all columns except the last one
                train_array[:, -1],   # All rows, only the last column
                test_array[:, :-1],   # All rows, all columns except the last one
                test_array[:, -1]    # All rows, only the last column
            )
            
            # --- Dictionary of all models to be tested ---
            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            # This dictionary will hold the reports
            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # --- Find the best model ---
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
            
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            # Check if the best model is good enough
            if best_model_score < 0.6:
                # Wrap your string in a base Exception() object
                raise CustomException(Exception("No best model found. All R2 scores are below 60%."))
        
        
            logging.info(f"Best model found on test dataset: {best_model_name}")
            logging.info(f"Best R2 Score: {best_model_score}")


            # Save the single best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test,predicted)
            
            return r2_square
            
        except Exception as e:
            raise CustomException(e)