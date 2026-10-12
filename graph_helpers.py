import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.var_level_and_volatility_utils import daily_tick_loss, exceedance_day_details, exceedance_day_summary


def _var_hover_data(var_df, residualbank):
    """var_df + the bank's predicted_variance / predicted_sigma / mu_hat / standardized_residual per date; returns (merged, hover_cols, hovertemplate)."""
    resid_df = residualbank.to_dataframe()

    merged = var_df.join(
        resid_df[['predicted_variance', 'predicted_sigma', 'mu_hat', 'standardized_residual']],
        how='left'
    )
    # var_df'te n_residuals_used varsa hover'da goster (yoksa eski davranis)
    has_n_used = 'n_residuals_used' in merged.columns
    hover_cols = ['predicted_variance', 'predicted_sigma', 'mu_hat', 'standardized_residual'] + (['n_residuals_used'] if has_n_used else [])
    hovertemplate = (
        '<b>%{x}</b><br>' +
        'VaR Forecast: %{y:.2%}<br>' +
        'predicted_variance (model_output): %{customdata[0]:.3e}<br>' +
        'predicted_sigma (sqrt of variance): %{customdata[1]:.2%}<br>' +
        'mu_hat: %{customdata[2]:.2%}<br>' +
        'Standardized Residual: %{customdata[3]:.2f}' +
        ('<br>Residuals used: %{customdata[4]:.0f}' if has_n_used else '') +
        '<extra></extra>'
    )
    return merged, hover_cols, hovertemplate


def _add_train_forecast_bands(fig, calendar, dates, **vrect_kwargs):
    """Coloured background bands for the train (blue) and forecast (orange) windows, clipped to the plotted `dates`."""
    dates = pd.DatetimeIndex(dates)
    for label, window, color, opacity in (
        ('train', calendar.train_data.index, 'steelblue', 0.10),
        ('forecast', calendar.forecast_dates, 'orange', 0.15),
    ):
        inside = window[window.isin(dates)]
        if len(inside):
            fig.add_vrect(x0=inside[0], x1=inside[-1], fillcolor=color, opacity=opacity, layer='below',
                          line_width=0, annotation_text=label, annotation_position='top left', **vrect_kwargs)


def _add_time_navigation(fig, dates):
    """
    Go to any time on the date axis: a range slider under the graph (drag or resize the window), quick-range
    buttons (1w / 1m / 3m / 6m / all) and a "go to month" dropdown. The y axis re-fits the visible window.
    """
    dates = pd.DatetimeIndex(dates)
    months = pd.period_range(dates.min(), dates.max(), freq='M')
    month_buttons = [dict(label='Go to month: all', method='relayout', args=[{'xaxis.autorange': True}])] + [
        dict(
            label=str(month), method='relayout',
            args=[{'xaxis.range': [month.start_time.strftime('%Y-%m-%d'), month.end_time.strftime('%Y-%m-%d')]}],
        )
        for month in months
    ]

    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07),
        rangeselector=dict(
            x=0, y=1.14, xanchor='left', yanchor='bottom',
            buttons=[
                dict(count=7, label='1w', step='day', stepmode='backward'),
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(step='all', label='all'),
            ],
        ),
    )
    fig.update_layout(
        updatemenus=[dict(
            type='dropdown', buttons=month_buttons, showactive=True,
            x=0.42, y=1.14, xanchor='left', yanchor='bottom',
        )],
        margin=dict(t=150),
    )


def _add_exceedance_markers(fig, merged):
    """
    Highlights the days with loss > varForecast (VaR exceedances) on the realized-loss line: a red ring plus the
    running number 1, 2, 3, ... in date order, so the count in the legend and the numbers on the graph agree.
    """
    exceeded = merged[merged['loss'] > merged['varForecast']]
    if exceeded.empty:
        return
    numbers = np.arange(1, len(exceeded) + 1)

    fig.add_trace(go.Scatter(
        x=exceeded.index,
        y=exceeded['loss'],
        mode='markers+text',
        text=numbers.astype(str),
        textposition='top center',
        textfont=dict(color='crimson', size=12),
        name=f'Exceedance (loss > VaR): {len(exceeded)}',
        marker=dict(symbol='circle-open', size=13, color='crimson', line=dict(width=2)),
        customdata=np.column_stack([numbers, exceeded['varForecast'].to_numpy(dtype=float)]),
        hovertemplate=(
            '<b>%{x}</b><br>Exceedance #%{customdata[0]}<br>'
            'Realized Loss: %{y:.2%}<br>VaR Forecast: %{customdata[1]:.2%}<extra></extra>'
        ),
    ))


def plot_var_vs_loss(var_df, residualbank, title='VaR Forecast vs Realized Loss', calendar=None, mark_exceedances=True,
                     time_navigation=True):
    """
    VaR Forecast ve Realized Loss'u plotly ile çizer.
    Hover'da predicted_variance (model_output), predicted_sigma, mu_hat ve standardized_residual gösterilir.
    mark_exceedances=True ise loss > varForecast olan günler kırmızı halkayla vurgulanır ve tarih sırasına göre
    numaralanır (1, 2, 3, ...); legend'da toplam sayı yazar.

    Parameters
    ----------
    var_df : pd.DataFrame
        Index'i Date olan, 'varForecast' ve 'loss' kolonlarını içeren dataframe.
    residualbank : ResidualBank
        .to_dataframe() metoduyla predicted_variance, mu_hat, standardized_residual
        kolonlarını Date index'iyle döndüren obje.
    title : str
        Grafik başlığı.
    calendar : GroundTruthCalendar, optional
        Verilirse train ve forecast dönemleri arka plan bandı olarak (mavi / turuncu) gösterilir.
    mark_exceedances : bool
        loss > varForecast günlerini vurgula ve numarala.
    time_navigation : bool
        True ise istedigin zamana gitmek icin range slider, 1w/1m/3m/6m/all butonlari ve "ay sec" dropdown'i eklenir.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    merged, hover_cols, hovertemplate = _var_hover_data(var_df, residualbank)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged['varForecast'],
        mode='lines',
        name='VaR Forecast',
        line=dict(color='red', width=2),
        customdata=merged[hover_cols].values,
        hovertemplate=hovertemplate,
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

    if mark_exceedances:
        _add_exceedance_markers(fig, merged)

    if calendar is not None:
        _add_train_forecast_bands(fig, calendar, merged.index)

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Loss/Var Prediction (pozit)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis_tickformat='.1%'
    )

    if time_navigation:
        _add_time_navigation(fig, merged.index)

    return fig


def plot_var_with_window_size(var_df, residualbank, title='VaR Forecast vs Realized Loss', calendar=None):
    """
    Ust panel: plot_var_vs_loss ile ayni grafik (VaR Forecast + Realized Loss).
    Alt panel: her VaR gunu icin quantile'in kac residual ile hesaplandigi
    (var_df['n_residuals_used']). Rolling pencerede sabit N, expanding pencerede
    (multi_day_var_calculation_window_size=None) gun gun artan bir cizgi gorursun.

    Parameters
    ----------
    var_df : pd.DataFrame
        'varForecast', 'loss' ve 'n_residuals_used' kolonlarini iceren, Date indexli dataframe
        (calendar.var_from_residual_bank_* ciktisi).
    residualbank : ResidualBank
        Hover bilgisi (predicted_sigma, mu_hat, standardized_residual) icin.
    title : str
        Grafik basligi.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if 'n_residuals_used' not in var_df.columns:
        raise KeyError("var_df has no 'n_residuals_used' column; produce it with calendar.var_from_residual_bank_* first.")

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

    if calendar is not None:
        _add_train_forecast_bands(fig, calendar, var_df.index, row='all', col='all')

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


def plot_var_models_vs_loss(runs, title='VaR Forecast vs Realized Loss (models)', calendar=None, time_navigation=True):
    """
    plot_var_vs_loss for several models of the SAME calendar: one VaR line per model on top of the one
    realized-loss line. Hover per model shows its predicted_variance (model_output), mu_hat, standardized
    residual and residuals used, like plot_var_vs_loss.

    Parameters
    ----------
    runs : dict[str, tuple[pd.DataFrame, ResidualBank]]
        {model_name: (var_df, residualbank)}, var_df = calendar.var_from_residual_bank_*(...) of that model.
        All runs must come from the same calendar (their 'loss' columns must agree).
    title : str
    calendar : GroundTruthCalendar, optional
        Verilirse train ve forecast dönemleri arka plan bandı olarak (mavi / turuncu) gösterilir.
    time_navigation : bool
        True ise range slider, 1w/1m/3m/6m/all butonlari ve "ay sec" dropdown'i eklenir.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not runs:
        raise ValueError("runs is empty.")

    losses = pd.concat({name: var_df['loss'] for name, (var_df, _) in runs.items()}, axis=1)
    loss = losses.bfill(axis=1).iloc[:, 0]
    if losses.sub(loss, axis=0).abs().max().max() > 1e-12:
        raise ValueError("The runs disagree on the realized loss: they must all come from the same calendar.")

    colors = pc.qualitative.Plotly
    fig = go.Figure()
    for i, (name, (var_df, residualbank)) in enumerate(runs.items()):
        merged, hover_cols, hovertemplate = _var_hover_data(var_df, residualbank)
        fig.add_trace(go.Scatter(
            x=merged.index,
            y=merged['varForecast'],
            mode='lines',
            name=f'VaR: {name}',
            line=dict(color=colors[i % len(colors)], width=2),
            customdata=merged[hover_cols].values,
            hovertemplate=hovertemplate,
        ))

    fig.add_trace(go.Scatter(
        x=loss.index,
        y=loss,
        mode='lines+markers',
        name='Realized Loss',
        line=dict(color='steelblue', width=1),
        marker=dict(size=4),
        hovertemplate='<b>%{x}</b><br>Realized Loss: %{y:.2%}<extra></extra>'
    ))

    if calendar is not None:
        _add_train_forecast_bands(fig, calendar, loss.index)

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Loss/Var Prediction (pozit)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis_tickformat='.1%'
    )

    if time_navigation:
        _add_time_navigation(fig, loss.index)

    return fig


def plot_models_output_vs_realized_variance(calendar, residualbanks, ewma_lam=0.94, ewma_min_periods=30,
                                            only_forecast_window=True,
                                            title='Model output (predicted_variance) vs EWMA realized variance'):
    """
    Only the variances, no VaR: every model's predicted_variance (model_output) against the calendar's
    EWMA realized variance (calendar.ewma_realized_variance: the variance of day d contains r_d itself).

    Parameters
    ----------
    calendar : GroundTruthCalendar
        The calendar all the models were run on; it provides the realized variance.
    residualbanks : dict[str, ResidualBank]
        {model_name: residualbank}, the bank returned by calculate_multi_day_standart_residuals_from_*.
    ewma_lam, ewma_min_periods :
        Settings of the realized-variance bank (RiskMetrics decay, seed bars).
    only_forecast_window : bool
        True: only the forecast dates; False: every date the banks hold (train in-sample + forecast).
    title : str

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not residualbanks:
        raise ValueError("residualbanks is empty.")

    realized = calendar.ewma_realized_variance(ewma_lam, ewma_min_periods)
    keep = (lambda s: s.loc[s.index.isin(calendar.forecast_dates)]) if only_forecast_window else (lambda s: s)

    # variance of a daily return is ~1e-5 (unreadable on an axis): shown in %^2, i.e. (return in %)^2 = variance * 1e4
    to_pct_sq = 1e4

    colors = pc.qualitative.Plotly
    fig = go.Figure()
    for i, (name, residualbank) in enumerate(residualbanks.items()):
        predicted = keep(residualbank.to_dataframe()['predicted_variance'].astype(float)) * to_pct_sq
        fig.add_trace(go.Scatter(
            x=predicted.index,
            y=predicted,
            mode='lines',
            name=f'predicted_variance (model_output): {name}',
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='<b>%{x}</b><br>' + name + ': %{y:.3f} %²<extra></extra>',
        ))

    realized = keep(realized) * to_pct_sq
    fig.add_trace(go.Scatter(
        x=realized.index,
        y=realized,
        mode='lines',
        name=f'EWMA realized variance (lam={ewma_lam})',
        line=dict(color='black', width=2, dash='dash'),
        hovertemplate='<b>%{x}</b><br>EWMA realized variance: %{y:.3f} %²<extra></extra>',
    ))

    _add_train_forecast_bands(fig, calendar, realized.index)

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Variance (%², return in percent squared)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis_tickformat='.2f'
    )
    return fig


def plot_model_output_vs_realized_variance(calendar, residualbank, model_name='model', **kwargs):
    """Single-model plot_models_output_vs_realized_variance (same parameters, see there)."""
    return plot_models_output_vs_realized_variance(calendar, {model_name: residualbank}, **kwargs)


def _windowed_figure(fig, dates, start_date, window_bars):
    """
    A copy of `fig` showing `window_bars` bars from `start_date` (None = until the last date). The start is the
    first bar on/after start_date (clamped to the data); the y axis is fitted to the points inside the window.
    """
    dates = pd.DatetimeIndex(dates)
    start_pos = min(int(dates.searchsorted(pd.Timestamp(start_date))), len(dates) - 1)
    end_pos = len(dates) - 1 if window_bars is None else min(start_pos + window_bars - 1, len(dates) - 1)
    start, end = dates[start_pos], dates[end_pos]

    windowed = go.Figure(fig)
    windowed.update_xaxes(range=[start - pd.Timedelta(days=1), end + pd.Timedelta(days=1)])

    lows, highs = [], []
    for trace in windowed.data:
        if getattr(trace, 'x', None) is None or getattr(trace, 'y', None) is None:
            continue
        x = pd.DatetimeIndex(trace.x)
        y = np.asarray(trace.y, dtype=float)
        inside = (x >= start) & (x <= end) & ~np.isnan(y)
        if inside.any():
            lows.append(y[inside].min())
            highs.append(y[inside].max())
    if lows:
        low, high = min(lows), max(highs)
        pad = (high - low) * 0.12 or 0.001
        windowed.update_yaxes(range=[low - pad, high + pad])
    return windowed


def _date_picker_view(fig, dates, window_options):
    """Date picker + window length dropdown above `fig`; every change re-draws the figure (ipywidgets)."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError("The date picker needs ipywidgets: pip install ipywidgets") from exc

    dates = pd.DatetimeIndex(dates)
    picker = widgets.DatePicker(
        description='Start date', value=dates[0].date(), min=dates[0].date(), max=dates[-1].date(),
    )
    window = widgets.Dropdown(description='Window', options=window_options, value=None)
    out = widgets.Output()

    def redraw(*_):
        start_date = picker.value if picker.value is not None else dates[0].date()
        with out:
            out.clear_output(wait=True)
            _windowed_figure(fig, dates, start_date, window.value).show()

    picker.observe(redraw, names='value')
    window.observe(redraw, names='value')
    redraw()
    return widgets.VBox([widgets.HBox([picker, window]), out])


_DATE_PICKER_WINDOWS = [
    ('all bars', None), ('1 week (5 bars)', 5), ('1 month (21 bars)', 21),
    ('3 months (63 bars)', 63), ('6 months (126 bars)', 126),
]


def var_vs_loss_date_picker(var_df, residualbank, title='VaR Forecast vs Realized Loss', calendar=None,
                            mark_exceedances=True, window_options=_DATE_PICKER_WINDOWS):
    """
    plot_var_vs_loss with a DatePicker: pick a start date and a window length, the graph jumps there
    (needs ipywidgets; display the returned widget as the last expression of a cell).
    """
    fig = plot_var_vs_loss(var_df, residualbank, title=title, calendar=calendar,
                           mark_exceedances=mark_exceedances, time_navigation=False)
    return _date_picker_view(fig, var_df.index, window_options)


def var_models_vs_loss_date_picker(runs, title='VaR Forecast vs Realized Loss (models)', calendar=None,
                                   window_options=_DATE_PICKER_WINDOWS):
    """plot_var_models_vs_loss with a DatePicker and a window length dropdown (see var_vs_loss_date_picker)."""
    fig = plot_var_models_vs_loss(runs, title=title, calendar=calendar, time_navigation=False)
    first_var_df = next(iter(runs.values()))[0]
    return _date_picker_view(fig, first_var_df.index, window_options)


def plot_exceedance_days_focus(var_dfs, days, window_bars=10, row_height=330):
    """
    ONE figure with the given days stacked one under another. Per day a row with
      - a graph (left): the VaR line of every model and the realized loss for `window_bars` bars around the day,
        the day marked with a dotted line, the days some model missed ringed with the number of models that
        missed them;
      - a table (right): per model the loss, the VaR, the difference loss - VaR and whether the VaR was
        exceeded (exceeded rows tinted red).
    The same numbers as a plain table: dm.exceedance_day_details(var_dfs, days).

    Parameters
    ----------
    var_dfs : dict[str, pd.DataFrame]
        {model_name: var_df} of the SAME period (same dates and realized loss).
    days : pd.DataFrame | pd.DatetimeIndex | list
        The days to focus on, e.g. the dm.exceedance_days_by_model(...) table (its index is used).
    window_bars : int
        Bars shown on each side of the day.
    row_height : int
        Pixels per day. The figure is `row_height * number of days` tall: pass fewer days to get a shorter one.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not var_dfs:
        raise ValueError("var_dfs is empty.")
    if isinstance(days, pd.DataFrame):
        days = days.index
    days = pd.DatetimeIndex(days)

    names = list(var_dfs)
    dates = var_dfs[names[0]].index
    for name, var_df in var_dfs.items():
        if not var_df.index.equals(dates):
            raise ValueError(f"var_dfs[{name!r}] is on different dates than var_dfs[{names[0]!r}].")
    missing = days.difference(dates)
    if len(missing):
        raise KeyError(f"days not in var_dfs: {[d.date().isoformat() for d in missing]}")

    colors = pc.qualitative.Plotly
    n_days = len(days)
    losses = var_dfs[names[0]]['loss'].astype(float)

    # every number shown below is computed by the helpers (core.var_level_and_volatility_utils), not here
    summary = exceedance_day_summary(var_dfs, days)                       # per day, across the models
    details = exceedance_day_details(var_dfs, days, labels=False)         # per (day, model)

    # one title per grid cell, row-major: (graph, table) for every day; the table cell has no title
    titles = []
    for day in days:
        s = summary.loc[day]
        titles += [
            f"<b>{day.date()}</b>  |  loss {s['loss']:.2%}  |  missed by {int(s['n_missed'])} of {int(s['n_models'])} models<br>"
            f"diff (loss - VaR): avg {s['avg_difference']:+.2%}  |  min {s['min_difference']:+.2%}  |  max {s['max_difference']:+.2%}",
            '',
        ]

    height = row_height * n_days + 190
    # ~70 px between two days (room for the day title); the rest of the row is the graph / table
    vertical_spacing = min(70 / (height - 270), 0.9 / max(n_days - 1, 1))

    fig = make_subplots(
        rows=n_days, cols=2, column_widths=[0.58, 0.42],
        specs=[[{'type': 'xy'}, {'type': 'table'}]] * n_days,
        subplot_titles=titles,
        vertical_spacing=vertical_spacing, horizontal_spacing=0.03,
    )

    day_tables = []
    for row, day in enumerate(days, start=1):
        pos = dates.get_loc(day)
        window = dates[max(pos - window_bars, 0): pos + window_bars + 1]
        first = row == 1

        # graph: every model's VaR line and the one realized loss line
        for i, name in enumerate(names):
            part = var_dfs[name].loc[window]
            fig.add_trace(go.Scatter(
                x=part.index, y=part['varForecast'], mode='lines', name=name, legendgroup=name, showlegend=first,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>%{{x}}</b><br>{name} VaR: %{{y:.2%}}<extra></extra>',
            ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=window, y=losses.loc[window], mode='lines+markers', name='Realized Loss', legendgroup='loss',
            showlegend=first, line=dict(color='black', width=2), marker=dict(size=5),
            hovertemplate='<b>%{x}</b><br>Realized Loss: %{y:.2%}<extra></extra>',
        ), row=row, col=1)

        # the days in the window that at least one model missed, ringed, with the number of models that missed them
        window_summary = exceedance_day_summary(var_dfs, window)
        missed_by = window_summary['n_missed'][window_summary['n_missed'] > 0]
        if len(missed_by):
            fig.add_trace(go.Scatter(
                x=missed_by.index, y=window_summary.loc[missed_by.index, 'loss'], mode='markers+text',
                text=missed_by.astype(str),
                textposition='top center', textfont=dict(color='crimson', size=10),
                marker=dict(symbol='circle-open', size=11, color='crimson', line=dict(width=2)),
                name='Missed (n = models)', legendgroup='missed', showlegend=first,
                hovertemplate='<b>%{x}</b><br>missed by %{text} models<extra></extra>',
            ), row=row, col=1)
        fig.add_vline(x=day, line_dash='dot', line_color='gray', row=row, col=1)

        # table: per model the loss, the VaR and the difference on the day (exceeded rows tinted red)
        rows = details.loc[day]                                           # one row per model, in `names` order
        tint = ['mistyrose' if hit else 'white' for hit in rows['exceeded']]
        ink = ['crimson' if hit else 'black' for hit in rows['exceeded']]
        # (added after the loop: add_vline cannot look at a figure that already holds Table traces)
        day_tables.append((row, go.Table(
            columnwidth=[1.0, 0.8, 0.8, 0.9, 1.5],
            header=dict(values=['model', 'loss', 'VaR', 'diff', 'status'],
                        fill_color='lightgray', align='left', font=dict(size=10), height=22),
            cells=dict(
                values=[list(rows.index), [f'{v:.2%}' for v in rows['loss']], [f'{v:.2%}' for v in rows['var_forecast']],
                        [f'{v:+.2%}' for v in rows['difference']],
                        ['exceeded' if hit else 'not exceeded' for hit in rows['exceeded']]],
                fill_color=[tint] * 5, font=dict(color=[ink] * 5, size=10), align='left', height=22,
            ),
        )))

    for row, table in day_tables:
        fig.add_trace(table, row=row, col=2)

    # paper coordinates are fractions of the plot area: pixel offsets are converted with the figure height
    fig.update_layout(
        title=dict(text=f'Exceedance days, +/- {window_bars} bars around each day (all models)',
                   x=0, xanchor='left', y=1 - 18 / height, yanchor='top', font=dict(size=16)),
        template='plotly_white', height=height, margin=dict(t=170, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.0 + 45 / (height - 250), xanchor='left', x=0,
                    font=dict(size=11)),
    )
    fig.update_annotations(font_size=12)              # the day titles
    fig.update_yaxes(tickformat='.1%', tickfont=dict(size=10))
    fig.update_xaxes(tickformat='%b %d', tickfont=dict(size=10))
    return fig


def plot_cumulative_tick_loss_difference(var_dfs, baseline, var_confidence,
                                         title='Cumulative tick loss difference vs baseline'):
    """
    Where does each run gain or lose against a baseline run (e.g. a GARCH)? For every run the running sum of
    (its daily tick loss - the baseline's daily tick loss): a line going DOWN means the run is better than the
    baseline in that stretch, UP means worse. The baseline is the flat zero line.

    var_dfs : dict[str, pd.DataFrame]
        {run_name: var_df} on the same dates (as dm.compare_to_baseline takes them).
    """
    if baseline not in var_dfs:
        raise KeyError(f"baseline {baseline!r} is not in var_dfs: {list(var_dfs)}")
    base_tick = daily_tick_loss(var_dfs[baseline], var_confidence)

    colors = pc.qualitative.Plotly
    fig = go.Figure()
    fig.add_hline(y=0, line_dash='dot', line_color='gray', annotation_text=f'baseline: {baseline}',
                  annotation_position='top left')
    for i, (name, var_df) in enumerate(var_dfs.items()):
        if name == baseline:
            continue
        cumulative = (daily_tick_loss(var_df, var_confidence) - base_tick).cumsum()
        fig.add_trace(go.Scatter(
            x=cumulative.index, y=cumulative, mode='lines', name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='<b>%{x}</b><br>' + name + ': %{y:.4f}<extra></extra>',
        ))

    fig.update_layout(
        title=title, xaxis_title='Date', yaxis_title='cumulative tick loss difference (below 0 = better than baseline)',
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def plot_level_and_volatility(level_volatility, title='VaR level and volatility by run'):
    """
    Small-multiples view of the level-and-volatility table (columns = runs, rows = result fields, as built from
    dm.level_and_volatility_forecast_period): every measure on its own axis, runs side by side.
    Bar colour: blue = a run name ending in '_long', orange = '_wide' (anything else grey).

    Panels: tick loss (stacked: VaR too high + VaR too low, total = mean_tick_loss), conservativeness
    (VaR / realized quantile, 1 = the window's own quantile), coverage (exceedance rate vs expected),
    VaR volatility (coefficient of variation), jumpiness (mean |daily change| / mean VaR), tail severity.
    """
    m = level_volatility.astype(float)
    runs = list(m.columns)
    colors = {'long': '#1f77b4', 'wide': '#ff7f0e'}
    bar_colors = [colors.get(run.rsplit('_', 1)[-1], 'gray') for run in runs]

    fig = make_subplots(
        rows=2, cols=3, vertical_spacing=0.22, horizontal_spacing=0.07,
        subplot_titles=(
            'Mean tick loss (x 1e-4): lower is better',
            'Conservativeness: mean VaR / realized quantile',
            'Exceedance rate (dashed = expected)',
            'VaR volatility: std / mean',
            'VaR jumpiness: mean |daily change| / mean VaR',
            'Tail severity: mean size of exceedances',
        ),
    )

    # tick loss: the two parts stacked (overlay + base), the total on top is mean_tick_loss
    overshoot = m.loc['tick_loss_overshoot_part'] * 1e4
    exceedance = m.loc['tick_loss_exceedance_part'] * 1e4
    fig.add_trace(go.Bar(
        x=runs, y=overshoot, name='VaR too high (overshoot)', marker_color='lightgray',
        hovertemplate='%{x}<br>overshoot part: %{y:.2f}<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=runs, y=exceedance, base=overshoot, name='VaR too low (exceedances)', marker_color='crimson',
        text=[f'{t:.2f}' for t in overshoot + exceedance], textposition='outside',
        hovertemplate='%{x}<br>exceedance part: %{y:.2f}<extra></extra>',
    ), row=1, col=1)

    single_panels = (
        ('var_to_realized_quantile_ratio', 1, 2, '%{y:.2f}', 1.0),
        ('exceedance_rate', 1, 3, '%{y:.1%}', float(m.loc['expected_exceedance_rate'].iloc[0])),
        ('var_cv', 2, 1, '%{y:.2f}', None),
        ('relative_mean_abs_daily_change', 2, 2, '%{y:.2f}', None),
        ('mean_exceedance_size', 2, 3, '%{y:.2%}', None),
    )
    for field, row, col, fmt, reference in single_panels:
        fig.add_trace(go.Bar(
            x=runs, y=m.loc[field], marker_color=bar_colors, showlegend=False,
            texttemplate=fmt, textposition='outside',
            hovertemplate='%{x}<br>' + field + ': ' + fmt + '<extra></extra>',
        ), row=row, col=col)
        if reference is not None:
            fig.add_hline(y=reference, line_dash='dash', line_color='black', row=row, col=col)

    fig.update_layout(
        title=title, template='plotly_white', barmode='overlay', height=720,
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='left', x=0),
    )
    fig.update_yaxes(rangemode='tozero')
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
