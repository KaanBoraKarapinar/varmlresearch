import numpy as np
import pandas as pd
import plotly.colors as pc
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
        (calendar.var_from_residual_bank ciktisi).
    residualbank : ResidualBank
        Hover bilgisi (predicted_sigma, mu_hat, standardized_residual) icin.
    title : str
        Grafik basligi.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if 'n_residuals_used' not in var_df.columns:
        raise KeyError("var_df has no 'n_residuals_used' column; produce it with calendar.var_from_residual_bank first.")

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


def plot_ewma_bank(ewma_bank, split_date=None, heatmap_date=None, as_volatility=False,
                   title='EWMA Variance Bank'):
    """
    EwmaVarianceBank'in portfolio_variance ve cov_by_date icerigini tek figurde cizer.

    Sol taraf (zaman serileri):
      1) Portfoy varyansi (portfolio_variance), bank'in asil ciktisi.
      2) Hisse basina varyans: EWMA kovaryansin kosegeni (duz cizgi) ve MODELIN tahmini
         (sigma_diag, kesikli cizgi). Portfoy varyansinin koseger kismi modelden, korelasyon
         kismi EWMA'dan geldigi icin ikisi birlikte gorulur.
      3) Hisse ciftleri arasindaki EWMA korelasyonu (cov_by_date'ten turetilir).
    Sag taraf: heatmap_date gunundeki EWMA korelasyon matrisi.

    Parameters
    ----------
    ewma_bank : EwmaVarianceBank
        Doldurulmus bank (egitim sonrasi model.ewma_variance_bank_in_sample ya da forecast sonrasi kopya).
    split_date : pd.Timestamp | str | None
        Verilirse sol paneller uzerinde dikey kesikli cizgi (orn. egitim/forecast siniri,
        calendar.train_data_last_index[0]).
    heatmap_date : pd.Timestamp | str | None
        Korelasyon matrisinin gosterilecegi gun. None ise bank'in son gunu.
    as_volatility : bool
        False: varyanslar oldugu gibi. True: karekoku alinip % olarak (gunluk volatilite) gosterilir
        (panel 1 ve 2). Korelasyon paneli etkilenmez.
    title : str
        Grafik basligi.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    dates = pd.DatetimeIndex(sorted(ewma_bank.cov_by_date))
    if len(dates) == 0:
        raise ValueError("ewma_bank has no rows yet (cov_by_date is empty); train/fill it first.")
    tickers = list(ewma_bank.tickers)
    n = len(tickers)

    covs = np.stack([ewma_bank.cov_by_date[d].to_numpy() for d in dates])   # (T, n, n)
    ewma_var = np.einsum('tii->ti', covs)                               # EWMA'nin kendi kosegeni
    sd = np.sqrt(ewma_var)
    model_var = ewma_bank.sigma_diag.astype(float).loc[dates, tickers].to_numpy()
    portfolio_var = ewma_bank.portfolio_variance.astype(float).loc[dates].to_numpy()

    if as_volatility:
        scale, to_plot = 100.0, np.sqrt
        unit, fmt = 'volatilite (%)', '%{y:.3f}%'
    else:
        scale, to_plot = 1.0, (lambda x: x)
        unit, fmt = 'varyans', '%{y:.3e}'

    if heatmap_date is None:
        heatmap_date = dates[-1]
    heatmap_date = pd.Timestamp(heatmap_date)
    if heatmap_date not in ewma_bank.cov_by_date:
        raise KeyError(f"heatmap_date {heatmap_date} is not in ewma_bank.cov_by_date "
                       f"(range {dates[0].date()} .. {dates[-1].date()}).")
    cov_h = ewma_bank.cov_by_date[heatmap_date].to_numpy()
    sd_h = np.sqrt(np.diag(cov_h))
    corr_h = cov_h / np.outer(sd_h, sd_h)

    ticker_colors = pc.qualitative.Plotly
    pair_colors = pc.qualitative.Dark24

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {'rowspan': 3}], [{}, None], [{}, None]],
        column_widths=[0.68, 0.32], row_heights=[0.28, 0.40, 0.32],
        shared_xaxes='columns', vertical_spacing=0.07, horizontal_spacing=0.09,
        # subplot_titles follows the grid row by row: (1,1) portfoy, (1,2) heatmap, (2,1) hisse basina, (3,1) ciftler
        subplot_titles=(f'Portfoy {unit}', f'EWMA korelasyon matrisi ({heatmap_date.date()})',
                        f'Hisse basina {unit}: EWMA (duz) vs model (kesikli)', 'Hisse ciftleri arasi EWMA korelasyonu'),
    )

    # 1) portfoy varyansi
    fig.add_trace(go.Scatter(
        x=dates, y=to_plot(portfolio_var) * scale, mode='lines', name=f'Portfoy {unit}',
        line=dict(color='black', width=2),
        hovertemplate=f'<b>%{{x}}</b><br>Portfoy: {fmt}<extra></extra>',
    ), row=1, col=1)

    # 2) hisse basina: EWMA kosegeni (duz) ve modelin tahmini (kesikli)
    for i, t in enumerate(tickers):
        color = ticker_colors[i % len(ticker_colors)]
        fig.add_trace(go.Scatter(
            x=dates, y=to_plot(ewma_var[:, i]) * scale, mode='lines', name=f'{t} EWMA',
            legendgroup=t, line=dict(color=color, width=1.5),
            hovertemplate=f'<b>%{{x}}</b><br>{t} EWMA: {fmt}<extra></extra>',
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=to_plot(model_var[:, i]) * scale, mode='lines', name=f'{t} model',
            legendgroup=t, line=dict(color=color, width=1.5, dash='dash'),
            hovertemplate=f'<b>%{{x}}</b><br>{t} model: {fmt}<extra></extra>',
        ), row=2, col=1)

    # 3) ciftler arasi korelasyon (cov_by_date'ten)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr_ij = covs[:, i, j] / (sd[:, i] * sd[:, j])
            fig.add_trace(go.Scatter(
                x=dates, y=corr_ij, mode='lines', name=f'{tickers[i]}-{tickers[j]}',
                line=dict(color=pair_colors[k % len(pair_colors)], width=1.5),
                hovertemplate=f'<b>%{{x}}</b><br>{tickers[i]}-{tickers[j]}: %{{y:.2f}}<extra></extra>',
            ), row=3, col=1)
            k += 1

    # sag: korelasyon matrisi (heatmap_date)
    fig.add_trace(go.Heatmap(
        z=corr_h, x=tickers, y=tickers, zmin=-1, zmax=1, colorscale='RdBu', reversescale=True,
        text=np.round(corr_h, 2), texttemplate='%{text}', showscale=True,
        colorbar=dict(title='corr', len=0.6), name='korelasyon',
        hovertemplate='%{y} - %{x}: %{z:.3f}<extra></extra>',
    ), row=1, col=2)
    fig.update_yaxes(autorange='reversed', row=1, col=2)

    if split_date is not None:
        for r in (1, 2, 3):
            fig.add_vline(x=pd.Timestamp(split_date), row=r, col=1, line_dash='dot', line_color='gray')

    fig.update_layout(
        title=title, template='plotly_white', height=820, hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='left', x=0),
    )
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text=unit, row=2, col=1)
    fig.update_yaxes(title_text='korelasyon', range=[-1, 1], row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=1)

    return fig
