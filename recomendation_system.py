from flask import Flask, request, jsonify
import logging as lg
from datetime import datetime
import pandas as pd

from src.helpers import helpers
from main import main
method_dict, features = main()

app = Flask(__name__)
# logging app starting
lg.info(f"Application have started by: {str(datetime.now())}")


# init 404 catcher
@app.errorhandler(404)
def internal_server_error404(*args):
    return jsonify({'error': 'Bed request for a server!!!'}), 404


# init 500 catcher
@app.errorhandler(500)
def internal_server_error500(*args):
    return jsonify({'error': 'We have an issue!!!'}), 500


# start app
with app.app_context():

    @app.route('/recomendation', methods=['POST'])
    def create_task():
        """
        API endpoint to provide recomendations by user properties and recomendation method.

        Request Body:
        - UserID: str
            Username - just for clarifying.
        - Method: str
            Recomendation method.
        - Data: dict
            Params for recomendation
        Returns:
        - dict:-+
            A JSON response containing a number of a game.
        """
        # Getting the task details from the request body
        request_data = request.get_json()
        if request_data:
            if request_data["Method"] in method_dict.keys():
                pandas_df = helpers.change_str_to_int_values(pd.DataFrame(request_data["Data"], index=1))[features]
                result = method_dict["Method"](pandas_df)
                if request_data["Method"] == "mnn":
                    result = round(result * 50)
                # logging a work process
                lg.info(f"Recomendation: Method={request_data['Method']}: "
                        f"UserID={request_data['UserID']}: Result={result}: Datetime_={str(datetime.now())}")
                # Returning the created task as a JSON response
                return jsonify({"UseID": request_data['UserID'], "Result": result, "Datetime": str(datetime.now())}), 201
        else:
            # Error message, task was now found
            lg.error(f'Something went wrong!!!')
            # Returning a 404 error if the task is not found
            return jsonify({'error': 'Something went wrong!!!'}), 404


if __name__ == '__main__':
    app.run(debug=True)
