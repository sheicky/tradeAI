import pandas as pd
import plotly.graph_objects as go


def create_gauge_chart(probability):
    """
    Crée un graphique de type gauge pour afficher la probabilité de hausse du marché
    """

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "rgba(50,150,255,0.8)"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "rgba(255,0,0,0.1)"},
                    {"range": [30, 70], "color": "rgba(255,255,0,0.1)"},
                    {"range": [70, 100], "color": "rgba(0,255,0,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "Market Trend Probability", "font": {"size": 24}},
            number={"suffix": "%", "font": {"size": 20}},
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400
    )

    return fig


def create_model_probability_chart(probabilities):
    """
    Crée un graphique en barres pour comparer les probabilités des différents modèles
    """
    models = list(probabilities.keys())
    probs = [probabilities[model] * 100 for model in models]

    colors = [
        "rgba(50,150,255,0.8)" if p > 50 else "rgba(255,50,50,0.8)" for p in probs
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=models,
                y=probs,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in probs],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Model Confidence Levels",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        },
        yaxis_title="Probability (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", range=[0, 100]),
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
    )

    return fig


def create_historical_chart(df, date, window=30):
    """
    Crée un graphique historique des indicateurs clés
    """
    # Convertir la colonne date en datetime si ce n'est pas déjà fait
    df["Data"] = pd.to_datetime(df["Data"])

    # Sélectionner une fenêtre de données autour de la date
    date = pd.to_datetime(date)
    mask = (df["Data"] >= date - pd.Timedelta(days=window)) & (
        df["Data"] <= date + pd.Timedelta(days=window)
    )
    df_window = df[mask]

    fig = go.Figure()

    # Ajouter les indicateurs clés
    key_indicators = {
        "XAU BGNL": "Gold Price",
        "VIX": "Volatility Index",
        "LUMSTRUU": "Global Bond Index",
        "LMBITR": "Market Index",
        "LUACTRUU": "Aggregate Bond Index",
    }

    for col, name in key_indicators.items():
        if col in df_window.columns:
            # Normaliser les valeurs pour une meilleure visualisation
            values = df_window[col]
            normalized = (values - values.min()) / (values.max() - values.min())

            fig.add_trace(
                go.Scatter(x=df_window["Data"], y=normalized, name=name, mode="lines")
            )

    fig.update_layout(
        title="Historical Market Indicators",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )

    return fig
