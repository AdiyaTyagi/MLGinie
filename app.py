import os
import pandas as pd
import numpy as np
import dash
import base64
import pickle
import time
from dash import html, dcc, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objs as go
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import io

latest_feature_importance = None
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=["https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap"])
app.title = "AutoML Insight Dashboard"

# Apply base style
base_style = {
    "fontFamily": "'Poppins', sans-serif",
    "backgroundColor": "#DCFFB7",
    "padding": "20px"
}

header_style = {
    "textAlign": "center",
    "fontWeight": "700",
    "fontSize": "32px",
    "marginBottom": "30px",
    "color": "#1e1e1e"
}

button_style = {
    'margin': '10px',
    'padding': '12px 24px',
    'border': 'none',
    'borderRadius': '10px',
    'fontWeight': '500',
    'fontSize': '16px',
    'cursor': 'pointer'
}

footer_style = {
    'backgroundColor': '#2d2d2d',
    'color': '#fff',
    'padding': '40px 20px',
    'textAlign': 'center',
    'marginTop': '60px'
}

# Chart layout theme
chart_layout = go.Layout(
    paper_bgcolor='#DCFFB7',
    plot_bgcolor='#ffffff',
    font=dict(family="'Poppins', sans-serif", size=14, color="#1e1e1e"),
    margin=dict(l=40, r=40, t=40, b=40),
    hovermode='closest',
    showlegend=True,
    title=dict(x=0.5, font=dict(size=20))
)

# Layout
app.layout = html.Div(style=base_style, children=[
    html.H2("AutoML Insight Dashboard", style=header_style),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['📂 Drag and Drop or ', html.A('Select CSV File')]),
        style={
            'width': '60%', 'margin': 'auto', 'padding': '40px',
            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '12px',
            'textAlign': 'center', 'backgroundColor': '#fff', 'color': '#7f8c8d'
        },
        multiple=False
    ),

    html.Div([
        html.Button("🔧 Run Hyperparameter Tuning", id='tune-button', n_clicks=0,
                    style={**button_style, 'backgroundColor': '#27ae60', 'color': 'white'}),

        html.Button("🔁 Reset to Base Models", id='reset-button', n_clicks=0,
                    style={**button_style, 'backgroundColor': '#c0392b', 'color': 'white'})
    ], style={'textAlign': 'center'}),

    html.Div([
        html.Button("⬇️ Download Best Model", id='download-button', n_clicks=0,
                    style={**button_style, 'backgroundColor': '#2980b9', 'color': 'white'}),

        html.Button("⬇️ Download Results CSV", id='download-results-button', n_clicks=0,
                    style={**button_style, 'backgroundColor': '#8e44ad', 'color': 'white'})
    ], style={'textAlign': 'center'}),

    dcc.Download(id="download-model"),
    dcc.Download(id="download-results"),

    html.Div(id='file-uploaded-msg', style={'textAlign': 'center', 'marginTop': '20px', 'fontWeight': '500'}),
    html.Div(id='dataset-summary', style={'margin': '20px auto', 'width': '90%'}),

    dcc.Loading(
        id="loading", type="circle",
        children=[
            html.Div(id='auto-results', style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '18px'}),
            dcc.Graph(id='model-metrics-chart', config={'displayModeBar': False}),
            html.Div(id='model-table-container'),
            dcc.Graph(id='feature-importance-graph', config={'displayModeBar': False}),
            dcc.Graph(id='confusion-matrix', config={'displayModeBar': False})
        ]
    ),

    html.Div([
        html.H3("Conclusion", style={'textAlign': 'center', 'marginTop': '60px', 'color': '#1e1e1e'}),
        html.P("This AutoML dashboard empowers users to easily upload datasets, explore feature importance, evaluate various ML models, and download the best performing model or results—all through a clean, intuitive interface.",
               style={'textAlign': 'center', 'maxWidth': '800px', 'margin': '0 auto', 'fontSize': '18px', 'color': '#333'})
    ]),

    html.Div([
        html.Div("We Can Connect Through:", style={'fontSize': '36px', 'fontWeight': '700', 'marginBottom': '10px'}),
    
        html.Div([
             dcc.Dropdown(
                options=[
                    {"label": "Priyanshu", "value": "priyanshtmr27@gmail.com"},
                    {"label": "Adiya", "value": "adiyatyagi2@gmail.com"}
                ],
                id='Email-dropdown',
                placeholder="📧 Email",
                style={'width': '40%', 'margin': '10px auto', 'color': '#000000', 'cursor': 'pointer'}
            ),
            html.Div(id='selected-email-display', style={
                'textAlign': 'center',
                'marginTop': '10px'
            }),
            dcc.Dropdown(
                options=[
                    {"label": "Priyanshu", "value": "https://www.linkedin.com/in/tomar-priyanshu/"},
                    {"label": "Adiya", "value": "https://www.linkedin.com/in/adiyatyagi/"}
                ],
                id='linkedin-dropdown',
                placeholder="LinkedIn",
                style={'width': '40%', 'margin': '10px auto', 'color': '#000000'}
            ),
            dcc.Dropdown(
                options=[
                    {"label": "Priyanshu", "value": "https://github.com/priyanshu2706-oss"},
                    {"label": "Adiya", "value": "https://github.com/priyanshu2706-oss"} 
                ],
                id='github-dropdown',
                placeholder="GitHub",
                style={'width': '40%', 'margin': '10px auto', 'color': '#000000'}
            )
        ], style={'marginTop': '20px'})
    ], style=footer_style)
])

# Add redirect callbacks

@app.callback(
    Output('selected-email-display', 'children'),
    Input('Email-dropdown', 'value'),
    prevent_initial_call=True
)
def display_selected_email(email):
    if email:
        return html.Div(email, style={
            'backgroundColor': '#ffffff',
            'padding': '8px 16px',
            'borderRadius': '12px',
            'border': '1px solid #ccc',
            'display': 'inline-block',
            'cursor': 'copy',
            'userSelect': 'all',  # So user can easily copy it
            'fontWeight': '500',
            'color': '#1e1e1e'
        })
    return None

@app.callback(
    Output('linkedin-dropdown', 'value'),
    Input('linkedin-dropdown', 'value'),
    prevent_initial_call=True
)
def open_linkedin(value):
    if value:
        import webbrowser
        webbrowser.open_new_tab(value)
    return None

@app.callback(
    Output('github-dropdown', 'value'),
    Input('github-dropdown', 'value'),
    prevent_initial_call=True
)
def open_github(value):
    if value:
        import webbrowser
        webbrowser.open_new_tab(value)
    return None

def preprocess_data(df):
    df = df.copy()
    df.dropna(inplace=True)
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, label_encoders, df


def train_models(X_train, X_test, y_train, y_test, feature_names, tuning=False):
    global best_model_global, latest_confusion_matrix, latest_feature_importance
    results = []
    best_f1 = 0
    feature_importance_data = []

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    param_distributions = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Decision Tree': {'max_depth': [3, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
        'SVM': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
        'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 6]}
    }

    target_type = type_of_target(y_train)
    is_multiclass = target_type == "multiclass"

    for name, model in models.items():
        start = time.time()
        if tuning:
            clf = RandomizedSearchCV(model, param_distributions[name], n_iter=5, cv=3,
                                     scoring='f1_weighted' if is_multiclass else 'f1', n_jobs=-1)
            clf.fit(X_train, y_train)
            model = clf.best_estimator_
            tuning_type = 'RandomizedSearchCV'
        else:
            model.fit(X_train, y_train)
            tuning_type = 'Base'
        end = time.time()

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        if y_prob is not None:
            if is_multiclass:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            else:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = 0

        f1 = f1_score(y_test, y_pred, average='weighted' if is_multiclass else 'binary')
        acc = accuracy_score(y_test, y_pred)
        training_time = round(end - start, 2)

        if f1 > best_f1:
            best_f1 = f1
            best_model_global = model
            latest_confusion_matrix = confusion_matrix(y_test, y_pred)

            if hasattr(model, 'feature_importances_'):
                latest_feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'Confusion Matrix': str(confusion_matrix(y_test, y_pred).tolist()),
            'Training Time (s)': training_time,
            'Tuning': tuning_type
        })

    return pd.DataFrame(results)


def recommend_best_model(df):
    return df.sort_values(by='F1 Score', ascending=False).iloc[0]


@app.callback(
    [Output('file-uploaded-msg', 'children'),
     Output('dataset-summary', 'children'),
     Output('auto-results', 'children'),
     Output('model-metrics-chart', 'figure'),
     Output('model-table-container', 'children'),
     Output('feature-importance-graph', 'figure'),
     Output('confusion-matrix', 'figure')],
    [Input('upload-data', 'contents'),
     Input('tune-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def process_upload(content, tune_clicks, reset_clicks, filename):
    global latest_results_df
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if content is None:
        return '', '', '', {}, '', {}, {}

    try:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        df = pd.read_csv(io.StringIO(decoded))

        X, y, encoders, full_df = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tuning = triggered_id == 'tune-button'
        results_df = train_models(X_train, X_test, y_train, y_test, full_df.columns[:-1], tuning=tuning)
        best_model = recommend_best_model(results_df)
        latest_results_df = results_df

        # Summary table
        summary_table = html.Div([
            html.H4("📊 Dataset Overview"),
            html.P(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"),
            dash_table.DataTable(
                columns=[
                    {"name": "Column", "id": "column"},
                    {"name": "Data Type", "id": "dtype"},
                    {"name": "Missing Values", "id": "missing"},
                    {"name": "Unique Values", "id": "unique"}
                ],
                data=[{
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing": df[col].isnull().sum(),
                    "unique": df[col].nunique()
                } for col in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#f7dc6f', 'fontWeight': 'bold'},
                page_size=10
            )
        ])

        bar_fig = px.bar(results_df, x='Model', y=['Accuracy', 'F1 Score', 'ROC AUC'],
                         barmode='group', title="Model Performance Comparison")

        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in results_df.columns],
            data=results_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'lightblue', 'fontWeight': 'bold'},
            page_size=10
        )

        feature_fig = px.bar(latest_feature_importance, x='Feature', y='Importance',
                             title="Feature Importances") if latest_feature_importance is not None else {}

        cm_fig = px.imshow(latest_confusion_matrix, text_auto=True,
                           title="Confusion Matrix", labels=dict(x="Predicted", y="Actual")) \
            if latest_confusion_matrix is not None else {}

        return (
            f"✅ File '{filename}' uploaded and processed successfully! ({'Tuned' if tuning else 'Base'})",
            summary_table,
            html.H4(f"Recommended Model: {best_model['Model']} (F1 Score: {best_model['F1 Score']:.4f})"),
            bar_fig,
            table,
            feature_fig,
            cm_fig
        )

    except Exception as e:
        return f"❌ Error processing file: {e}", '', '', {}, '', {}, {}


@app.callback(
    Output("download-model", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_best_model(n_clicks):
    if best_model_global is not None:
        model_bytes = pickle.dumps(best_model_global)
        return dcc.send_bytes(model_bytes, filename="best_model.pkl")
    return None


@app.callback(
    Output("download-results", "data"),
    Input("download-results-button", "n_clicks"),
    prevent_initial_call=True
)
def download_results_csv(n_clicks):
    global latest_results_df
    if not latest_results_df.empty:
        return dcc.send_data_frame(latest_results_df.to_csv, filename="model_results.csv")
    return None


# ✅ Expose the server for Gunicorn
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
