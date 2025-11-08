import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # --- THIS IS THE FIX ---
        # It should only take 'e' as an argument, not 'e, sys'
        raise CustomException(e)

from sklearn.metrics import r2_score
from src.exception import CustomException

def evaluate_models(X_train, X_test, y_test, y_train, models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)
            
            # Use model.predict, not models.predict
            y_test_pred = model.predict(X_test)
            
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        # Fix: CustomException only takes one argument
        raise CustomException(e)