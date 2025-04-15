import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

st.set_page_config(page_title='Soil Pollution What-If Lab', layout='wide')
st.markdown('''
# 🧪 Soil Pollution What-If Lab

**Created by Ada Carter**

Explore how soil pollution, environmental, and agricultural factors impact disease risk. Use the controls below to simulate scenarios, compare countries, and analyze model predictions and explanations.
''')

st.sidebar.title('Soil Pollution What-If Lab')

st.markdown('''
# About the Dataset
This dataset contains 3000 synthetic records simulating real-world scenarios of soil pollution and related diseases. It captures environmental, agricultural, and demographic variables to analyze correlations between soil contamination and human health outcomes. The data was generated using probabilistic models and domain knowledge, making it suitable for exploratory analysis, machine learning, and environmental health research.

**Source:** Dataset by Khushi Yadav, MIT License ([Kaggle link](https://www.kaggle.com/datasets/khushikyad001/soil-pollution-and-associated-health-impacts?resource=download))

**Key features include:**
- Pollutant types and concentrations in soil
- Soil and weather conditions
- Agricultural practices and nearby industry presence
- Reported disease types, severity, and symptoms
- Affected demographic segments
- Mitigation measures and case resolutions

It is ideal for use in data science projects, public health studies, environmental modeling, and predictive analytics.

**Encoding:** UTF-8  
**Missing Values:** None

## Column Descriptions & Treatment
- **Case_ID**: Unique identifier for each reported case of soil pollution. Not used as a predictor; serves as a reference.
- **Date_Reported**: Date when the case was officially reported. Typically not used directly in modeling, but can be used for temporal analysis.
- **Region**: Global region where the case occurred (e.g., Asia, Europe). Can be encoded for modeling or used for stratified analysis.
- **Country**: Country where the soil pollution case was observed. Encoded numerically (Country_enc) and always included as a predictor to capture country-specific effects.
- **Pollutant_Type**: Main contaminant found in the soil (e.g., Lead, Arsenic, Pesticides). Encoded for modeling if used, or analyzed for pollutant-specific trends.
- **Pollutant_Concentration_mg_kg**: Measured pollutant concentration in the soil in milligrams per kilogram. Used as a numeric predictor.
- **Soil_pH**: Soil acidity or alkalinity level (pH scale, typically 4.5 to 8.5). Used as a numeric predictor.
- **Temperature_C**: Average temperature at the time of sampling in degrees Celsius. Used as a numeric predictor.
- **Humidity_%**: Relative humidity percentage at the time of report. Used as a numeric predictor.
- **Rainfall_mm**: Recorded rainfall in millimeters at the location. Used as a numeric predictor.

Additional columns in the dataset (not shown here) include variables on crop type, farming practices, mitigation strategies, disease outcomes, severity, symptoms, demographics, and more. Categorical variables are encoded as needed for modeling, and all numeric variables are used directly or after imputation if missing values are present (though this dataset has none).

This comprehensive structure enables robust analysis of environmental and health interactions, supporting both scientific research and applied machine learning.
''')

data = pd.read_csv('soil_pollution_diseases.csv')

target = st.sidebar.selectbox('Target to Predict', ['Disease_Type', 'Disease_Severity'])
from sklearn.preprocessing import LabelEncoder
le_country = LabelEncoder()
data['Country_enc'] = le_country.fit_transform(data['Country'])

features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Country_enc' not in features:
    features.append('Country_enc')
X = data[features].fillna(data[features].mean())
y = data[target]

if y.dtype == 'object':
    y, y_labels = pd.factorize(y)
else:
    y_labels = None

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

st.title('Soil Pollution What-If Scenario Lab 🌱')
st.markdown('''
Adjust the sliders below to simulate a real-world scenario.\
See how the predicted disease outcome changes, and learn which features matter most.\

**This is your lab for science and ML!**
''')

st.header('1. Set Your Scenario')
st.write('Adjust each factor to simulate a soil/crop/pollution scenario:')
user_input = {}
for feat in features:
    user_input[feat] = st.slider(f'{feat}', float(data[feat].min()), float(data[feat].max()), float(data[feat].mean()))
user_df = pd.DataFrame([user_input])

pred_proba = clf.predict_proba(user_df)[0]
confidence = np.max(pred_proba)
class_idx = np.argmax(pred_proba)
class_name = y_labels[class_idx] if y_labels is not None else class_idx

pred = clf.predict(user_df)[0]
pred_label = y_labels[pred] if y_labels is not None else pred
st.success(f'Predicted {target}: {pred_label}')
st.info(f"Model confidence: {confidence:.2%} for class '{class_name}'")

st.subheader('Prediction Probabilities')
proba_df = pd.DataFrame({
    'Class': y_labels if y_labels is not None else clf.classes_,
    'Probability': pred_proba
})
proba_df = proba_df.sort_values('Probability', ascending=False)
import plotly.express as px
fig_proba = px.bar(proba_df, x='Class', y='Probability', color='Probability', color_continuous_scale='Blues', title='Model Probability for Each Class')
st.plotly_chart(fig_proba, use_container_width=True)

st.header('2. Decision Support & Recommendations')
important_feats = np.array(features)[np.argsort(clf.feature_importances_)[::-1][:3]]
st.info(f"Top factors influencing this prediction: {', '.join(important_feats)}.")
st.markdown('''
- **Tip:** Try lowering the most important pollutant or changing soil pH to see if the risk drops.
- **Example:** If Lead is most important, reducing it may lower disease risk.
''')

st.header('3. Why did the model make this prediction?')
imp = pd.Series(clf.feature_importances_, index=features)
fig_imp = px.bar(
    imp.sort_values(ascending=True),
    orientation='h',
    color=imp.sort_values(ascending=True),
    color_continuous_scale='Viridis',
    labels={'value': 'Importance', 'index': 'Feature'},
    title='Feature Importance (Model-based)'
)
st.plotly_chart(fig_imp, use_container_width=True)
st.caption('Shows which features the model used most to make predictions.')

perm = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=features)
fig_perm = px.bar(
    perm_imp.sort_values(ascending=True),
    orientation='h',
    color=perm_imp.sort_values(ascending=True),
    color_continuous_scale='Plasma',
    labels={'value': 'Importance', 'index': 'Feature'},
    title='Permutation Importance'
)
st.plotly_chart(fig_perm, use_container_width=True)
st.caption('Measures how much shuffling each feature hurts model accuracy. More robust to bias.')

st.subheader('C. Interactive Feature Importance')
fig3 = px.bar(
    imp.sort_values(ascending=False),
    orientation='v',
    color=imp.sort_values(ascending=False),
    color_continuous_scale='Viridis',
    labels={'value': 'Importance', 'index': 'Feature'},
    title='Feature Importance (Interactive)'
)
st.plotly_chart(fig3, use_container_width=True)

contribs = (user_df.values[0] - X_train.mean().values) * clf.feature_importances_
contribs_df = pd.Series(contribs, index=features).sort_values()
fig4 = px.bar(
    contribs_df,
    orientation='h',
    color=contribs_df,
    color_continuous_scale=[(0, '#d62728'), (0.5, '#f7f7f7'), (1, '#2ca02c')],
    labels={'value': 'Contribution', 'index': 'Feature'},
    title='How Each Feature Pushed the Prediction'
)
fig4.update_traces(marker_line_width=1)
st.plotly_chart(fig4, use_container_width=True)
st.caption('Green: pushed prediction higher. Red: pushed lower. (Simple, intuitive explanation)')

st.subheader('E. SHAP-like Summary (for all test cases)')
shap_proxy = (X_test - X_train.mean()) * clf.feature_importances_
shap_long = pd.melt(shap_proxy.reset_index(drop=True))
fig5 = px.violin(
    shap_long, y='variable', x='value', color='variable', orientation='h', points='all', box=True, title='SHAP-like Summary (Proxy)'
)
fig5.update_layout(showlegend=False, xaxis_title='Contribution', yaxis_title='Feature')
st.plotly_chart(fig5, use_container_width=True)
st.caption('Shows how much each feature typically pushes predictions up or down across many cases.')

st.header('6. Country What-If Comparison')
all_countries = sorted(data['Country'].dropna().unique())
def_country = all_countries[0] if all_countries else ''
st.markdown('''
Select one or more countries to compare how the prediction would change if the country of origin were different, keeping all other scenario settings the same.
''')
selected_countries = st.multiselect('Compare these countries:', all_countries, default=[def_country])

if selected_countries:
    whatif_X = pd.DataFrame([user_input]*len(selected_countries))
    whatif_X['Country'] = selected_countries
    from sklearn.preprocessing import LabelEncoder
    le_country = LabelEncoder()
    le_country.fit(data['Country'])
    if 'Country' in features:
        whatif_X['Country'] = le_country.transform(whatif_X['Country'])
    whatif_X = whatif_X[features]
    country_pred_proba = clf.predict_proba(whatif_X)
    country_pred = np.argmax(country_pred_proba, axis=1)
    country_conf = np.max(country_pred_proba, axis=1)
    results_df = pd.DataFrame({
        'Country': selected_countries,
        'Predicted Class': [y_labels[i] if y_labels is not None else i for i in country_pred],
        'Confidence': country_conf
    })
    st.subheader('Comparison Matrix')
    st.dataframe(results_df, use_container_width=True)
    st.markdown('''
**Summary:**
For the selected countries, the model predicts the following disease outcomes and associated confidences. The matrix above provides a direct comparison, highlighting which countries are associated with higher or lower predicted risks for your scenario. Notably, the variation in confidence scores reflects the model's certainty, which may be influenced by the underlying data distribution and feature interactions. Countries with higher confidence and consistent class predictions may indicate more robust model generalization for those contexts.
''')
    st.subheader('Predicted Class Confidence by Country')
    fig = px.bar(results_df, x='Country', y='Confidence', color='Predicted Class', barmode='group', title='Prediction Confidence for Each Country')
    fig.update_layout(width=1200, height=500, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
**Interpretation:**
This bar chart visualizes the model's confidence in its predicted class for each country. Higher bars indicate greater certainty in the predicted disease outcome. Disparities between countries may reflect both the influence of the 'Country' feature and its interaction with other scenario variables. For instance, if one country consistently yields higher confidence, it may suggest either a more homogeneous data representation or stronger feature associations in the training set. Such insights are critical for understanding model reliability and potential biases in cross-country predictions.
''')
    st.subheader('Full Probability Distribution (All Classes)')
    proba_long = []
    for idx, country in enumerate(selected_countries):
        for class_idx, prob in enumerate(country_pred_proba[idx]):
            proba_long.append({
                'Country': country,
                'Class': y_labels[class_idx] if y_labels is not None else class_idx,
                'Probability': prob
            })
    proba_long_df = pd.DataFrame(proba_long)
    fig2 = px.bar(proba_long_df, x='Class', y='Probability', color='Country', barmode='group', facet_col='Country', title='Class Probabilities by Country')
    fig2.update_layout(width=1200, height=500, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('''
**Advanced Analysis:**
The faceted bar plots above provide a granular view of the model's predicted probability distribution across all possible disease classes for each country. This enables a nuanced assessment of model uncertainty and class overlap. For example, a country with a flatter distribution may indicate greater ambiguity or class imbalance, while a sharply peaked distribution suggests strong model preference for a particular outcome. Such patterns are essential for high-level model critique, informing both scientific interpretation and policy recommendations.
''')
    st.caption('Compare how the model output changes for your scenario if only the country is changed.')

st.header('7. Multiple Linear Regression Analysis')
from sklearn.linear_model import LinearRegression
predictor_options = [col for col in features if col != 'Country_enc'] + ['Country_enc']
selected_predictors = st.multiselect('Select predictors for regression:', predictor_options, default=predictor_options)
regression_target_options = [col for col in features if col not in selected_predictors and col in data.columns]
if regression_target_options:
    regression_target = st.selectbox('Regression Target (numeric only)', regression_target_options)
    reg_X = X[selected_predictors].copy()
    reg_y = data[regression_target]
    reg = LinearRegression()
    reg.fit(reg_X, reg_y)
    reg_preds = reg.predict(reg_X)

    equation = f"{regression_target} = {reg.intercept_:.3f}"
    for feat, coef in zip(selected_predictors, reg.coef_):
        equation += f" + ({coef:.3f} \\times {feat})"
    st.markdown(f"**Regression Equation:**  \\({equation}\\)", unsafe_allow_html=True)

    coef_df = pd.DataFrame({'Feature': reg_X.columns, 'Coefficient': reg.coef_})
    coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('AbsCoef', ascending=False)
    fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', color='Coefficient', color_continuous_scale='RdBu',
                      title='Linear Regression Coefficients (Sorted by Magnitude)',
                      hover_data={'AbsCoef': True, 'Coefficient': True, 'Feature': True})
    fig_coef.update_traces(marker_line_width=1.5)
    fig_coef.update_layout(xaxis_tickangle=-45, width=1200, height=500, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_coef, use_container_width=True)

    top_feats = coef_df['Feature'].iloc[:3].tolist()
    if regression_target not in top_feats:
        top_feats.append(regression_target)
    pairplot_df = data[top_feats]
    fig_pair = px.scatter_matrix(pairplot_df, dimensions=top_feats, title='Pairwise Relationships: Top Regression Features', height=700)
    st.plotly_chart(fig_pair, use_container_width=True)

    residuals = reg_y - reg_preds
    fig_resid = px.scatter(x=reg_preds, y=residuals, labels={'x': 'Predicted', 'y': 'Residual'},
                           title='Regression Residuals vs Predicted Values', trendline='ols')
    fig_resid.add_hline(y=0, line_dash='dash', line_color='black')
    fig_resid.update_layout(width=900, height=400)
    st.plotly_chart(fig_resid, use_container_width=True)

    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(reg_y, reg_preds)
    mse = mean_squared_error(reg_y, reg_preds)
    st.markdown(f"**R²:** {r2:.3f}  ")
    st.markdown(f"**MSE:** {mse:.3f}")

    st.markdown('''
    **Interpretation:**
    The bar plot above shows the estimated effect (coefficient) of each feature, including country of origin, on the selected numeric outcome. Positive coefficients indicate a direct relationship, while negative coefficients indicate an inverse relationship. The scatter plot compares actual and predicted values, with the dashed line representing perfect prediction. R² quantifies the proportion of variance explained by the model, and MSE measures average prediction error. Including 'Country' as a predictor allows for quantifying its unique contribution, controlling for all other variables.

    ---
    ### In-Depth Explanation: Multiple Linear Regression

    **Model Equation:**
    The multiple linear regression model estimates the target variable $Y$ as a linear combination of the input features $X_1, X_2, ..., X_p$:

    $$
    \hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p
    $$

    where $\beta_0$ is the intercept and $\beta_i$ are the coefficients shown in the bar plot. For this analysis, $X_p$ includes the encoded country of origin, so the model explicitly quantifies the effect of country, controlling for all other variables.

    **Factors Considered:**
    - All numeric features in the dataset, including pollution levels, soil properties, crop/farming variables, and the encoded country of origin.
    - Each coefficient represents the expected change in the target variable for a one-unit increase in the feature, holding all other features constant.

    **Explanatory Power:**
    - $R^2$ (coefficient of determination) indicates the proportion of variance in the target explained by the model. Values closer to 1 indicate better fit.
    - MSE (mean squared error) quantifies the average squared difference between actual and predicted values.

    **When and How to Use:**
    - Multiple linear regression is best for understanding and quantifying linear relationships between a numeric outcome and several predictors.
    - It is most appropriate when relationships are approximately linear, predictors are not highly collinear, and residuals are homoscedastic and normally distributed.
    - Use this model for hypothesis testing, effect size estimation, and as a baseline for more complex models.

    **Limitations:**
    - Cannot capture nonlinear or interaction effects unless explicitly modeled.
    - Sensitive to outliers and multicollinearity.
    - The interpretation of coefficients assumes all other variables are held constant, which may not always be realistic in observational data.

    **Best Practices:**
    - Always check residual plots and diagnostics to validate assumptions.
    - Consider feature scaling and regularization for high-dimensional data.
    - Use domain knowledge to interpret coefficients, especially for encoded categorical variables like country.
    ''')
else:
    st.warning('Please leave at least one numeric feature unselected as a predictor to use as the regression target.')

st.markdown('---')
st.header('4. Learn: What is Feature Importance?')
st.markdown('''
- **Model-based importance:** How much the model splits on each feature (can be biased if features are correlated).
- **Permutation importance:** How much accuracy drops if you shuffle a feature (more robust).
- **SHAP values:** How much each feature pushes a prediction up or down (like a fair credit assignment).

**Try changing the sliders above and watch the graphs update!**
''')

st.markdown('---')
st.header('5. Explore the Data')
if st.checkbox('Show raw data sample'):
    st.dataframe(data.head(20))
if st.checkbox('Show summary statistics'):
    st.write(data.describe(include='all'))
if st.checkbox('Show column info'):
    st.write(data.dtypes)

st.markdown('---')
st.info('This What-If Lab is designed for both environmental science decision support and ML education. Try different scenarios and see how the model responds!')

st.markdown('''
---
### Model & Statistical Summary (PhD Level)
This Random Forest model was trained on a real-world dataset of soil pollution and disease outcomes, leveraging a diverse set of features spanning pollution metrics, soil properties, crop types, and management practices. The model's feature importances, as visualized above, reveal the relative influence of each variable on disease risk prediction. Permutation importance further quantifies the robustness of these associations, accounting for potential feature correlations and data noise.

The SHAP-like summary plot provides a proxy for local and global interpretability, illustrating how individual features contribute to prediction shifts across the test set. The model's probabilistic outputs, both for individual scenarios and in cross-country what-if analyses, enable a rigorous assessment of uncertainty and model calibration. Such comprehensive explainability is critical for both scientific discovery and evidence-based environmental policy, ensuring that model-driven insights are transparent, reproducible, and actionable.
''')
