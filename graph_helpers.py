import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_var_vs_loss(var_df, residualbank, title='VaR Forecast vs Realized Loss'):
    """
    VaR Forecast ve Realized Loss'u plotly ile çizer.
    Hover'da predicted_sigma, mu_hat ve standardized_residual gösterilir.

    Parameters
    ----------
    var_df : pd.DataFrame
        Index'i Date olan, 'varForecast' ve 'loss' kolonlarını içeren dataframe.
    residualbank : ResidualBank
        .to_dataframe() metoduyla predicted_sigma, mu_hat, standardized_residual
        kolonlarını Date index'iyle döndüren obje.
    title : str
        Grafik başlığı.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    resid_df = residualbank.to_dataframe()

    merged = var_df.join(
        resid_df[['predicted_sigma', 'mu_hat', 'standardized_residual']],
        how='left'
    )
    # var_df'te n_residuals_used varsa hover'da goster (yoksa eski davranis)
    has_n_used = 'n_residuals_used' in merged.columns
    hover_cols = ['predicted_sigma', 'mu_hat', 'standardized_residual'] + (['n_residuals_used'] if has_n_used else [])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged['varForecast'],
        mode='lines',
        name='VaR Forecast',
        line=dict(color='red', width=2),
        customdata=merged[hover_cols].values,
        hovertemplate=(
            '<b>%{x}</b><br>' +
            'VaR Forecast: %{y:.2%}<br>' +
            'Predicted Sigma: %{customdata[0]:.2%}<br>' +
            'mu_hat: %{customdata[1]:.2%}<br>' +
            'Standardized Residual: %{customdata[2]:.2f}' +
            ('<br>Residuals used: %{customdata[3]:.0f}' if has_n_used else '') +
            '<extra></extra>'
        )
    ))

    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged['loss'],
        mode='lines+markers',
        name='Realized Loss',
        line=dict(color='steelblue', width=1),
        marker=dict(size=4),
        hovertemplate='<b>%{x}</b><br>Realized Loss: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Loss/Var Prediction (pozit)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis_tickformat='.1%'
    )

    return fig


def plot_var_with_window_size(var_df, residualbank, title='VaR Forecast vs Realized Loss'):
    """
    Ust panel: plot_var_vs_loss ile ayni grafik (VaR Forecast + Realized Loss).
    Alt panel: her VaR gunu icin quantile'in kac residual ile hesaplandigi
    (var_df['n_residuals_used']). Rolling pencerede sabit N, expanding pencerede
    (multi_day_var_calculation_window_size=None) gun gun artan bir cizgi gorursun.

    Parameters
    ----------
    var_df : pd.DataFrame
        'varForecast', 'loss' ve 'n_residuals_used' kolonlarini iceren, Date indexli dataframe
        (multi_day_var_from_* fonksiyonlarinin ciktisi).
    residualbank : ResidualBank
        Hover bilgisi (predicted_sigma, mu_hat, standardized_residual) icin.
    title : str
        Grafik basligi.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if 'n_residuals_used' not in var_df.columns:
        raise KeyError("var_df has no 'n_residuals_used' column; produce it with multi_day_var_from_* first.")

    top = plot_var_vs_loss(var_df, residualbank, title=title)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.06,
    )
    for trace in top.data:
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=var_df.index,
        y=var_df['n_residuals_used'],
        mode='lines+markers',
        name='Residuals used',
        line=dict(color='gray', width=2, shape='hv'),
        marker=dict(size=4),
        hovertemplate='<b>%{x}</b><br>Residuals used: %{y}<extra></extra>',
    ), row=2, col=1)

    fig.update_layout(
        title=title,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text='Loss/Var Prediction (pozit)', tickformat='.1%', row=1, col=1)
    fig.update_yaxes(title_text='# residuals', rangemode='tozero', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    return fig
