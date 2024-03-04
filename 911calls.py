import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Initialize Plotly for offline mode
init_notebook_mode(connected=True)

# Load the dataset
df = pd.read_csv('911.csv')

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())

# Get the count of calls by ZIP code
print(df['zip'].value_counts().head())

# Get the count of calls by township
print(df['twp'].value_counts().head())

# Get the number of unique titles (reasons for calls)
print(df['title'].nunique())

# Extract the reason for each call from the title column
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

# Display the count of calls by reason
print(df['Reason'].value_counts())

# Create a count plot of calls by reason
sns.countplot(x=df['Reason'], data=df)
plt.show()

# Convert the timeStamp column to datetime format
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# Extract hour, month, and day of the week from timeStamp
df['Hour'] = df['timeStamp'].dt.hour
df['Month'] = df['timeStamp'].dt.month
df['Day of Week'] = df['timeStamp'].dt.day_name()

# Create a count plot of calls by day of the week, with hue by reason
sns.countplot(x=df['Day of Week'], hue='Reason', data=df, palette='viridis')
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
plt.show()

# Create a count plot of calls by month, with hue by reason
sns.countplot(x=df['Month'], hue='Reason', data=df, palette='viridis')
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
plt.show()

# Group the data by month and count the number of calls
byMonth = df.groupby(by='Month').count()
byMonth['twp'].plot()
plt.show()

# Create a linear regression plot of calls by month
sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())
plt.show()

# Extract date from timeStamp
df['Date'] = df['timeStamp'].dt.date

# Group the data by date and count the number of calls
graph = df.groupby('Date').count()
graph['twp'].plot()
plt.tight_layout()
plt.show()

# Plot the count of traffic-related calls by date
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
plt.show()

# Plot the count of fire-related calls by date
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()
plt.show()

# Plot the count of EMS-related calls by date
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
plt.show()

# Group the data by day of the week and hour and count the number of calls
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

# Create a heatmap of calls by day of the week and hour
sns.heatmap(dayHour, cmap='viridis')
plt.show()

# Create a clustered heatmap of calls by day of the week and hour
sns.clustermap(dayHour, cmap='viridis')
plt.show()

# Group the data by day of the week and month and count the number of calls
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

# Create a heatmap of calls by day of the week and month
sns.heatmap(dayMonth, cmap='viridis')
plt.show()

# Create a clustered heatmap of calls by day of the week and month
sns.clustermap(dayMonth, cmap='viridis')
plt.show()
