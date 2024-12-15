import numpy as np
import pickle
import streamlit as st
import warnings

# Load the saved models
failure_model = pickle.load(open("Machine Failure.pkl", 'rb'))
failure_type_model = pickle.load(open("failure_type_model.pkl", 'rb'))

def predict_failure_and_type(input_data):
    try:
        # Convert input_data to float and reshape
        input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)

        # Predict machine failure
        failure_prediction = failure_model.predict(input_data_as_numpy_array)

        # Predict failure type if machine failure is predicted
        if failure_prediction[0] == 1:
            failure_type = failure_type_model.predict(input_data_as_numpy_array)
        else:
            failure_type = "No Failure"

        return failure_prediction[0], failure_type, None  # Return results and no error message
    except Exception as e:
        return None, None, str(e)  # Return None for results and the error message

def validate_input(input_data):
    error_messages = {}

    # Define input field names for reference
    input_field_names = [
        'Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]

    for i, val in enumerate(input_data):
        field_name = input_field_names[i]

        if not val:
            error_messages[field_name] = f"{field_name} is missing."
        else:
            try:
                # Attempt to convert input value to float
                float_val = float(val)
                if float_val < 0:
                    error_messages[field_name] = f"{field_name} cannot be negative."
            except ValueError:
                error_messages[field_name] = f"Invalid value for {field_name}. Please enter a valid number."

    return error_messages

def main():
    st.title('Machine Failure Prediction Web App')

    # Collect input data from the user

    # Getting the input data from the user
    col1, col2 = st.columns(2)

    input_field_names = [
        'Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]

    input_data = {}

    for i, field_name in enumerate(input_field_names):
        with col1 if i % 2 == 0 else col2:
            input_data[field_name] = st.text_input(f"{i+1}. {field_name} value")

    # Create placeholders for displaying the results and error messages
    failure_prediction_result = ""
    failure_type_result = ""
    prediction_error_message = ""

    # Initialize error_messages with an empty dictionary
    error_messages = {}

    # Suppress the sklearn warning related to feature names
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # Check if the user has clicked the prediction button
    if st.button('Predict Machine Failure'):
        error_messages = validate_input(list(input_data.values()))

        if not error_messages:
            input_values = list(input_data.values())
            failure_prediction, failure_type, prediction_error = predict_failure_and_type(input_values)

            if prediction_error is not None:
                prediction_error_message = f"Error during prediction: {prediction_error}"
            else:
                failure_prediction_result = f"Machine Failure Prediction: {'Yes' if failure_prediction == 1 else 'No'}"
                failure_type_result = f"Predicted Failure Type: {failure_type}"

    # Display the results or error messages in the Streamlit app
    for field_name, error_message in error_messages.items():
        st.error(f"{field_name}: {error_message}")

    if prediction_error_message:
        st.error(prediction_error_message)
    else:
        st.success(failure_prediction_result)
        st.success(failure_type_result)

if __name__ == '__main__':
    main()

