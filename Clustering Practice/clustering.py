import numpy as np
import scipy as sp
import scipy.spatial.distance as spd
import scipy.cluster.hierarchy as sph
import sklearn as sk
import sklearn.cluster as skc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


set_colors_url = 'https://drive.google.com/uc?export=download&id=1zFi7sNXbl_mlPNnBSkkVNryQZqdcOCR5'
set_colors = pd.read_excel(set_colors_url)
set_colors.fillna(0,inplace=True)
set_colors.head()

# Names of the color columns in one array
color_column_names = set_colors.columns.values[4:]
color_column_names

# Over 10,000 sets and over 100 colors
set_colors.shape

points = alt.Chart(lego_colors).mark_point().encode(
    x="Number of Unique Pieces",
    y="Number of Unique Sets",
  
    tooltip=list(lego_colors.columns.values) #👀
  )

text = alt.Chart(lego_colors).mark_text(
    align="left",
    baseline="bottom"  
  ).encode(
    x="Number of Unique Pieces",
    y="Number of Unique Sets",
    text="Color"
  )

(points + text).properties(
    width=800
  ).interactive() #👀

dist_lego_color = spd.squareform(spd.pdist(lego_colors[['Number of Unique Pieces', 'Number of Unique Sets']], metric = 'euclidean'))
dist_lego_color.shape

lego_color_xy = sph.linkage(dist_lego_color, method='ward')  # obtain the linkage matrix
_ = sph.dendrogram(lego_color_xy)  # plot the linkage matrix as a dendrogram

plt.xlabel('Data Points')
plt.xticks(rotation=90)
plt.ylabel('Distance')
plt.suptitle('Dendrogram: Lego Colors', 
             fontweight='bold', fontsize=14);

lego_color_labels = sph.fcluster(lego_color_xy,12000, criterion='distance')
lego_colors['cluster_label'] = lego_color_labels
lego_colors

points = alt.Chart(lego_colors,title="Clustering of the Number of Unique Pieces & Sets ").mark_point().encode(
    x="Number of Unique Pieces",
    y="Number of Unique Sets",
    color = 'cluster_label:N',
   
  
  
)
text = alt.Chart(lego_colors).mark_text(
    align="left",
    baseline="bottom"  
  ).encode(
    x="Number of Unique Pieces",
    y="Number of Unique Sets",
    text="Color",
    color = 'cluster_label:N'
  )


(points + text).properties(
    height=600,
    width=800
    
  ).interactive() #👀
