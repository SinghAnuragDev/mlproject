import sys
from src.logger import logging

def error_message_detail(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_number = 0
    
    error_message = (
        "Error occurred in python script name [{0}] "
        "line number [{1}] error message [{2}]"
    ).format(file_name, line_number, str(error))
    
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: Exception):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)
    
    def __str__(self) -> str:
        return self.error_message

