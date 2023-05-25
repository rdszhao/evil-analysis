# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sns.set(font_scale=1.3)
# %%
df = pd.read_excel('data/social_data.xlsx', engine='openpyxl')
new_names = ['date', 'account', 'platform', 'campaign', 'impressions', 'engagements', 'media']
df = df.rename(columns=dict(zip(df.columns, new_names)))
df['rate'] = (df['engagements'] / df['impressions']).fillna(0)
platform_r = {'fbpage': 'fb', 'tiktok_business': 'tiktok', 'linkedin_company': 'linkedin'}
df['platform'] = df['platform'].str.lower().replace(platform_r)
df['account'] = df['account'].str.lower().str.strip()
df['campaign'] = df['campaign'].str.lower().str.strip()
df['day'] = df['date'].dt.day_name()
df['hour'] = df['date'].dt.hour
df = df[df['rate'] <= 1.0]
df = df.replace({'n/a': np.nan})
# %%
plot = sns.scatterplot(data=df[df['rate'] <= 1.0], x='date', y='rate', s=8).set(xticklabels=[])
# %%
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()
palette = {'>= 15%': 'green', '< 15%': 'grey'}
for platform, ax in zip(df['platform'].unique(), axes):
	dfd = df[df['platform'] == platform]
	cmap = np.where(dfd['rate'] >= 0.15, '>= 15%', '< 15%')
	plot = sns.scatterplot(data=dfd, x='date', y='rate', hue=cmap, palette=palette, ax=ax)
	axe = ax.get_legend().remove()
	axe = ax.set_title(f"engagement rates on {platform}")
	axe = ax.set_xticks([])
	axe = ax.set_yticks([])
handles, labels = axes[0].get_legend_handles_labels()
fig = fig.legend(handles, labels, loc=(0.46, 0.92))
# %%
df['rate'].mean()
(df['rate'] >= 0.15).sum() / df.shape[0]
# %%
# platform_rates = df.groupby('platform')['rate'].apply(lambda c: (c >= 0.15).sum() / len(c))
# platform_rates.plot.barh()
plot = sns.barplot(
    data=df,
    y='platform',
    x='rate',
    order=df.groupby('platform').mean()['rate'].sort_values().reset_index()['platform'],
    ci=None
).set(
    title='engagement rates on each platform',
    xlabel='average rate'
)
# %%
ordering = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plot = sns.barplot(
    data=df, x='day', y='rate', order=ordering, ci=None
).set(
    title='engagement rates throughout each day of the week',
    xlabel='average rate'
)
# %%
plot = sns.barplot(
    data=df, x='hour', y='rate', ci=None
).set(
    title='engagement rates throughout each hour of the day',
    xlabel='average rate'
)
# %%
games = ['csgo', 'dota2', 'valorant']
metrics = ['impressions', 'engagements', 'rate']
cmappings = (-0.5, 0.2), (0.2, -0.2), (-0.2, 0.2)
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for x_ax, metric, cmapping in zip(axes, metrics, cmappings):
	cmap = sns.cubehelix_palette(start=cmapping[0], rot=-cmapping[1], dark=0, light=0.7, as_cmap=True)
	for game, ax in zip(games, x_ax):
		dfd = df[df['account'] == game]
		plot = sns.scatterplot(data=dfd, x='date', y=metric, hue=metric, palette=cmap, ax=ax)
		norm = plt.Normalize(dfd[metric].min(), dfd[metric].max())
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		axe = ax.get_legend().remove()
		axe = ax.figure.colorbar(sm, ax=ax)
		axe = ax.set_title(f"{metric} on {game}")
		axe = sm.set_array([])
		axe = ax.set_xticks([])
		axe = ax.set_yticks([])

dfd = df[df['account'].isin(games)]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, metrics):
	plot = sns.barplot(data=dfd, y=metric, x= 'account', ci=None, ax=ax)
# %%
plot = sns.barplot(
    data=df, y='media', x='rate', order=df.groupby('media').mean()['rate'].sort_values().reset_index()['media'], ci=None
).set(
    title='engagement rates on each type of media',
    ylabel=None,
    xlabel='average rate'
)
# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=True)
axes = axes.flatten()

for platform, ax in zip(df['platform'].unique(), axes):
	dfd = df[df['platform'] == platform]
	plot = sns.barplot(data=dfd, y='media', x='rate', order=dfd.groupby('media').mean()['rate'].sort_values().reset_index()['media'], ci=None, ax=ax)
	axe = ax.set_title(f"engagement rates on {platform} across media")
	axe = ax.set(ylabel=None)
	axe = ax.set(xlabel='average rate')
# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

medias = df['media'].unique()
colors = ['cadetblue', 'grey', 'pink', 'blue', 'purple', 'red', 'green']
palette = dict(zip(medias, colors))

for platform, ax in zip(df['platform'].unique(), axes):
	dfd = df[df['platform'] == platform]
	plot = sns.scatterplot(data=dfd, x='date', y='rate', hue='media', palette=palette, ax=ax)
	axe = ax.get_legend().remove()
	axe = ax.set_title(f"engagement rates on {platform}")
	axe = ax.set_xticks([])
	axe = ax.set_yticks([])

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
labels, lines = zip(*dict(zip(labels, lines)).items())
fig = fig.legend(lines, labels, loc=(0.9, 0.4))
# %%
campaigns = df['campaign'].dropna().unique()
metrics = ['impressions', 'engagements', 'rate']
cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
cmappings = (-0.5, 0.2), (0.2, -0.2), (-0.2, 0.2)
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for x_ax, metric, cmapping in zip(axes, metrics, cmappings):
	cmap = sns.cubehelix_palette(start=cmapping[0], rot=-cmapping[1], dark=0, light=0.7, as_cmap=True)
	for campaign, ax in zip(campaigns, x_ax):
		dfd = df[df['campaign'] == campaign]
		plot = sns.scatterplot(data=dfd, x='date', y=metric, hue=metric, palette=cmap, ax=ax)
		norm = plt.Normalize(dfd[metric].min(), dfd[metric].max())
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		axe = ax.get_legend().remove()
		axe = ax.figure.colorbar(sm, ax=ax)
		axe = ax.set_title(f"{metric} on {campaign}")
		axe = sm.set_array([])
		axe = ax.set_xticks([])
		axe = ax.set_yticks([])

dfd = df[df['campaign'].isin(campaigns)]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, metrics):
	plot = sns.barplot(data=dfd, y=metric, x='campaign', ci=None, ax=ax).set(ylabel=f"average {metric}")
# %%
