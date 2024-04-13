#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

st.title('Исследование динамики зарплат в России')

df = pd.read_excel('C:/Users/dimas/Jupyter/Stepik/payments_ru.xlsx')
infl = pd.read_excel('C:/Users/dimas/Jupyter/Stepik/inflation.xlsx')
infl['inflation'] = infl['inflation'] / 100
#df.head()


df_melt = pd.melt(df, id_vars=['industry'], value_vars=df.columns[2:])
df_melt = df_melt.rename(columns={'variable':'year', 'value':'avg_payment'})
#df_melt.head()


df_melt = df_melt.merge(infl, on='year', how='left')
df_melt['avg_payment_plus_infl'] = round(df_melt['avg_payment'] * (1+df_melt['inflation']), 1)


df_melt_infl = pd.melt(df_melt, id_vars=['industry', 'year'], value_vars=['avg_payment', 'avg_payment_plus_infl'])
df_melt_infl['variable'] = df_melt_infl['variable'].replace({'avg_payment':'Средняя зарплата', 
                                                             'avg_payment_plus_infl':'Реальная зарплата'})
#df_melt_infl.head()


df_melt_sorted = df_melt.sort_values(by=['industry','year'])
df_melt_sorted = df_melt_sorted.set_index('year')
df_melt_sorted['lag'] = df_melt_sorted.groupby(['industry'])['avg_payment'].shift(1)
df_melt_sorted['payment_diff'] = df_melt_sorted['avg_payment'] - df_melt_sorted['lag']

#df_melt_sorted.head()

fig = px.line(df_melt_infl, x='year', y = 'value', color='variable', facet_col='industry', 
              title="Изменение фактических и реальных зарплат в РФ по годам",
              labels={
                     "value": "Зарплата, руб.",
                     "year": "Год",
                     "variable": "Тип зарплаты"   
                 })
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[0]))
#fig.show()
st.plotly_chart(fig)

fig = px.line(df_melt_sorted, x=df_melt_sorted.index, y = 'payment_diff', color='industry',
              title="Динамика разницы зарплат в сравнении с предыдущим годом",
              labels={
                     "payment_diff": "Разница, руб.",
                     "year": "Год",
                     "industry": "Направление"   
                 })
#fig.show()
st.plotly_chart(fig)

infl['inflation'] = infl['inflation']*100

fig = px.bar(infl, x='year', y='inflation',
             title="Изменение инфляции в РФ по годам",
             labels={
                     "inflation": "Инфляция, %",
                     "year": "Год"  
                 })
#fig.show()
st.plotly_chart(fig)

corr = df_melt_sorted.groupby('industry')[['payment_diff', 'inflation']].corr()
corr[::2]

df_melt_sorted_ = df_melt_sorted.sort_values(by='year')
df_melt_sorted_
