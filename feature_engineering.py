import pandas as pd


# ---------------------------------------------------------
# Feature engineering function
# Performs the same feature preparation used during
# model training and prediction.
# ---------------------------------------------------------
def prepare_features(data):

    # Create a copy so that the original DataFrame
    # is not modified.
    data = data.copy()


    # -----------------------------------------------------
    # Remove customerID because it is an identifier and
    # does not provide useful information for prediction.
    # -----------------------------------------------------
    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)


    # -----------------------------------------------------
    # Convert TotalCharges from string/object to numeric.
    # Invalid or blank values are converted to NaN.
    # -----------------------------------------------------
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )


    # -----------------------------------------------------
    # Replace missing TotalCharges values with 0.
    # In the original dataset, blank TotalCharges values
    # correspond to customers with zero tenure.
    # -----------------------------------------------------
    data["TotalCharges"] = data["TotalCharges"].fillna(0)


    # -----------------------------------------------------
    # Create ServiceCount.
    # Counts the number of additional services subscribed
    # to by each customer.
    # -----------------------------------------------------
    if "ServiceCount" not in data.columns:

        service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        data["ServiceCount"] = (
            data[service_columns] == "Yes"
        ).sum(axis=1)


    # -----------------------------------------------------
    # Create IsNewCustomer.
    # Customers with tenure of 12 months or less are
    # treated as new customers.
    #
    # 1 = New customer
    # 0 = Existing customer
    # -----------------------------------------------------
    if "IsNewCustomer" not in data.columns:

        data["IsNewCustomer"] = (
            data["tenure"] <= 12
        ).astype(int)


    # Return the cleaned data with engineered features.
    return data