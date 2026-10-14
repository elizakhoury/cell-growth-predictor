def run_workflow():
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    # Locate the dataset relative to this Python file.
    project_folder = Path(__file__).resolve().parents[1]
    dataset_path = (
        project_folder / "datasets" / "dataset_cell_growth_4factor.csv"
    )

    # Read the CSV file into a table.
    df = pd.read_csv(dataset_path)

    print("DATASET SIZE")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nCOLUMN NAMES")
    for column in df.columns:
        print(column)

    print("\nFIRST FIVE ROWS")
    print(df.head().to_string(index=False))

    print("\nDATA TYPES")
    print(df.dtypes.to_string())

    print("\nMISSING VALUES")
    print(df.isna().sum().to_string())

    print("\nDUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\nSUMMARY STATISTICS")
    print(df.describe().to_string())

    # Define the four input variables and the prediction target.
    feature_columns = [
        "Temperature [C]",
        "pH",
        "Glucose [mM]",
        "Lactate [mM]",
    ]
    target_column = "Specific Growth Rate [d^-1]"

    X = df[feature_columns]
    y = df[target_column]

    # Reserve 20% of the observations for final testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\nTRAINING AND TEST SETS")
    print(f"Training observations: {len(X_train)}")
    print(f"Test observations: {len(X_test)}")
    print(f"Input variables: {X_train.shape[1]}")

    # Use the same five validation splits for every model.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Measure prediction errors and explained variation.
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    #Establish a simple reference for the other models.
    baseline = DummyRegressor(strategy="mean")

    baseline_scores = cross_validate(
        baseline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
    )

    print("\nBASELINE: FIVE-FOLD CROSS-VALIDATION")
    print(f"Mean RMSE: {-baseline_scores['test_rmse'].mean():.4f} d^-1")
    print(f"Mean MAE: {-baseline_scores['test_mae'].mean():.4f} d^-1")
    print(f"Mean R²: {baseline_scores['test_r2'].mean():.4f}")

    # Scale the inputs, then fit a linear regression model.
    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression()),
    ])

    linear_scores = cross_validate(
        linear_model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
    )

    print("\nLINEAR REGRESSION: FIVE-FOLD CROSS-VALIDATION")
    print(f"Mean RMSE: {-linear_scores['test_rmse'].mean():.4f} d^-1")
    print(f"Mean MAE: {-linear_scores['test_mae'].mean():.4f} d^-1")
    print(f"Mean R²: {linear_scores['test_r2'].mean():.4f}")

    # Compare models that can represent curved relationships
    # and interactions between the four inputs.
    polynomial_results = []

    for degree in [2, 3]:
        polynomial_model = Pipeline([
            ("input_scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(
                degree=degree,
                include_bias=False,
            )),
            ("regression", LinearRegression()),
        ])

        polynomial_scores = cross_validate(
            polynomial_model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
        )

        train_rmse = -polynomial_scores["train_rmse"].mean()
        validation_rmse = -polynomial_scores["test_rmse"].mean()
        validation_mae = -polynomial_scores["test_mae"].mean()
        validation_r2 = polynomial_scores["test_r2"].mean()

        polynomial_results.append({
            "degree": degree,
            "train_rmse": train_rmse,
            "validation_rmse": validation_rmse,
            "validation_mae": validation_mae,
            "validation_r2": validation_r2,
        })

        print(f"\nPOLYNOMIAL REGRESSION: DEGREE {degree}")
        print(f"Mean training RMSE: {train_rmse:.4f} d^-1")
        print(f"Mean validation RMSE: {validation_rmse:.4f} d^-1")
        print(f"Mean validation MAE: {validation_mae:.4f} d^-1")
        print(f"Mean validation R²: {validation_r2:.4f}")

    # Compare different neighborhood sizes and weighting methods.
    knn_results = []

    for n_neighbors in [3, 5, 10, 15, 25]:
        for weights in ["uniform", "distance"]:
            knn_model = Pipeline([
                ("scaler", StandardScaler()),
                ("regression", KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric="euclidean",
                )),
            ])

            knn_scores = cross_validate(
                knn_model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
            )

            knn_results.append({
                "neighbors": n_neighbors,
                "weights": weights,
                "train_rmse": -knn_scores["train_rmse"].mean(),
                "validation_rmse": -knn_scores["test_rmse"].mean(),
                "validation_mae": -knn_scores["test_mae"].mean(),
                "validation_r2": knn_scores["test_r2"].mean(),
            })

    # Display the strongest validation results first.
    knn_results_df = pd.DataFrame(knn_results)
    knn_results_df = knn_results_df.sort_values("validation_rmse")

    print("\nKNN: FIVE-FOLD CROSS-VALIDATION")
    print(knn_results_df.to_string(index=False, float_format="%.4f"))

    # Collect the validation results for every model we compared.
    comparison_rows = [
        {
            "model": "Baseline (mean)",
            "validation_rmse": -baseline_scores["test_rmse"].mean(),
            "validation_mae": -baseline_scores["test_mae"].mean(),
            "validation_r2": baseline_scores["test_r2"].mean(),
        },
        {
            "model": "Linear regression",
            "validation_rmse": -linear_scores["test_rmse"].mean(),
            "validation_mae": -linear_scores["test_mae"].mean(),
            "validation_r2": linear_scores["test_r2"].mean(),
        },
    ]

    for result in polynomial_results:
        comparison_rows.append({
            "model": f"Polynomial degree {result['degree']}",
            "validation_rmse": result["validation_rmse"],
            "validation_mae": result["validation_mae"],
            "validation_r2": result["validation_r2"],
        })

    for result in knn_results:
        comparison_rows.append({
            "model": (
                f"KNN k={result['neighbors']}, "
                f"weights={result['weights']}"
            ),
            "validation_rmse": result["validation_rmse"],
            "validation_mae": result["validation_mae"],
            "validation_r2": result["validation_r2"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("validation_rmse")

    # Create the output folder if it does not already exist.
    tables_folder = project_folder / "tables"
    tables_folder.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(
        tables_folder / "model_comparison.csv",
        index=False,
        float_format="%.6f",
    )

    print("\nSaved tables/model_comparison.csv")
    print(f"Number of models compared: {len(comparison_df)}")

    # Fit the model selected using cross-validation.
    final_model = Pipeline([
        ("input_scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(
            degree=2,
            include_bias=False,
        )),
        ("regression", LinearRegression()),
    ])

    final_model.fit(X_train, y_train)

    # Evaluate on the reserved test observations.
    test_predictions = final_model.predict(X_test)

    test_rmse = root_mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    print("\nFINAL MODEL: DEGREE-2 POLYNOMIAL REGRESSION")
    print(f"Test observations: {len(y_test)}")
    print(f"Test RMSE: {test_rmse:.4f} d^-1")
    print(f"Test MAE: {test_mae:.4f} d^-1")
    print(f"Test R²: {test_r2:.4f}")

    # Save the overall test results.
    test_metrics = pd.DataFrame([{
        "model": "Polynomial degree 2",
        "test_observations": len(y_test),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }])

    test_metrics.to_csv(
        tables_folder / "test_metrics.csv",
        index=False,
    )

    # Save each test observation and its prediction.
    prediction_table = X_test.copy()
    prediction_table["actual_growth_rate"] = y_test
    prediction_table["predicted_growth_rate"] = test_predictions
    prediction_table["residual"] = y_test - test_predictions

    prediction_table.to_csv(
        tables_folder / "test_predictions.csv",
        index=False,
    )

    print("\nSaved test_metrics.csv and test_predictions.csv")

    # Create the folder for exported figures.
    figures_folder = project_folder / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compare actual and predicted growth rates.
    ax = axes[0]
    ax.scatter(
        y_test,
        test_predictions,
        color="tab:blue",
        alpha=0.7,
        edgecolors="black",
    )

    lower = min(y_test.min(), test_predictions.min())
    upper = max(y_test.max(), test_predictions.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        label="Perfect prediction",
    )

    ax.set_xlabel("Actual growth rate [d⁻¹]")
    ax.set_ylabel("Predicted growth rate [d⁻¹]")
    ax.set_title("Test set: actual versus predicted")
    ax.legend()

    # Show prediction errors.
    residuals = y_test - test_predictions

    ax = axes[1]
    ax.scatter(
        test_predictions,
        residuals,
        color="tab:orange",
        alpha=0.7,
        edgecolors="black",
    )

    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Predicted growth rate [d⁻¹]")
    ax.set_ylabel("Residual: actual − predicted [d⁻¹]")
    ax.set_title("Test set: residuals")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figures_folder / "test_predictions.png",
        dpi=300,
    )
    plt.close(fig)

    print("\nSaved figures/test_predictions.png")

    # Explore one input at a time using the fitted model.
    # Hold the other inputs at their training-data medians.
    reference_conditions = X_train.median()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    response_tables = []

    for ax, feature in zip(axes.flat, feature_columns):
        # Focus on the central 90% of training values.
        lower = X_train[feature].quantile(0.05)
        upper = X_train[feature].quantile(0.95)
        values = np.linspace(lower, upper, 100)

        conditions = pd.DataFrame(
            [reference_conditions.to_dict()] * len(values),
            columns=feature_columns,
        )
        conditions[feature] = values

        predicted_growth = final_model.predict(conditions)

        ax.plot(values, predicted_growth, color="tab:blue")
        ax.set_xlabel(feature)
        ax.set_ylabel("Predicted growth rate [d⁻¹]")
        ax.set_title(f"Varying {feature}")
        ax.grid(alpha=0.25)

        # Save the exact input combinations and predictions.
        response_table = conditions.copy()
        response_table.insert(0, "varied_feature", feature)
        response_table["predicted_growth_rate"] = predicted_growth
        response_tables.append(response_table)

    fig.suptitle(
        "Model responses with other inputs held at training medians"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        figures_folder / "feature_responses.png",
        dpi=300,
    )
    plt.close(fig)

    pd.concat(response_tables, ignore_index=True).to_csv(
        tables_folder / "feature_responses.csv",
        index=False,
    )

    print("\nREFERENCE CONDITIONS")
    print(reference_conditions.to_string())
    print("\nSaved feature_responses.png and feature_responses.csv")

    # Examine observed high- and low-growth examples
    # using only the training data.
    training_observations = X_train.copy()
    training_observations[target_column] = y_train

    high_growth = training_observations.nlargest(
        5, target_column
    ).copy()
    high_growth.insert(0, "group", "High observed growth")

    low_growth = training_observations.nsmallest(
        5, target_column
    ).copy()
    low_growth.insert(0, "group", "Low observed growth")

    growth_examples = pd.concat(
        [high_growth, low_growth],
        ignore_index=True,
    )

    growth_examples.to_csv(
        tables_folder / "growth_condition_examples.csv",
        index=False,
    )

    print("\nOBSERVED HIGH- AND LOW-GROWTH EXAMPLES")
    print(growth_examples.to_string(index=False))
    print("\nSaved tables/growth_condition_examples.csv")

    # Record the input ranges represented in the training data.
    training_ranges = pd.DataFrame({
        "minimum": X_train.min(),
        "5th_percentile": X_train.quantile(0.05),
        "median": X_train.median(),
        "95th_percentile": X_train.quantile(0.95),
        "maximum": X_train.max(),
    })

    training_ranges.index.name = "feature"

    training_ranges.to_csv(
        tables_folder / "training_input_ranges.csv"
    )

    print("\nTRAINING INPUT RANGES")
    print(training_ranges.to_string())
    print("\nSaved tables/training_input_ranges.csv")
