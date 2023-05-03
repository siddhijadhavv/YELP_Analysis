from itertools import product
import pandas as pd
import psycopg2
import streamlit as st
import plotly.express as px
#import pass_file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_data(cursor, query):
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    return df

st.title('Summary Dashboard')
#establishing the connection
conn = psycopg2.connect(database = "Yelp", user = "postgres", password = "root", host = "127.0.0.1", port = "5432")
print ("Opened database successfully")
#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Q1: What are the top 20 cities (with state information) with the most businesses on yelp?
top_20_cities_query = '''SELECT 
    state, city, COUNT(*) AS business_count
FROM
    yelp_business
GROUP BY state , city
ORDER BY business_count DESC
LIMIT 20;'''
#st.write("graphs")

top_20_cities = get_data(cursor, top_20_cities_query)

cursor.execute(top_20_cities_query)
data = cursor.fetchall()
column_names = ["state","city","count"]
top_20_cities = pd.DataFrame(data,columns = column_names)


print(top_20_cities)
#st.bar_chart(new[0],new[1])
fig = plt.figure(figsize=(10,5))
sns.barplot(data=top_20_cities, x="state", y="count")

#sns.barplot(new[0], new[1], alpha=0.8)


st.pyplot(fig)

class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels,colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')
import matplotlib.colors as mcolors
colors = []
for i in mcolors.CSS4_COLORS:
    colors.append(i)
    
top_20_cities['count'] = top_20_cities['count'].apply(pd.to_numeric)
top_20_cities['city'] = top_20_cities['city'].astype(str)
top_20_cities['state'] = top_20_cities['state'].astype(str)
import numpy as np
bubble_chart = BubbleChart(area=top_20_cities['count'], bubble_spacing=0.1)
bubble_chart.collapse()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
fig.set_size_inches(9, 13, forward=True)
bubble_chart.plot(ax, top_20_cities['state'],colors)
ax.axis("off")
ax.relim()
ax.autoscale_view()
st.pyplot(fig)

data = [10,20,30,40,50,60,70,80,90,100]
df_n = pd.DataFrame(data, columns=['Numbers'])

top_n = st.selectbox('Choose top "n" Cities', df_n)
top_n_cities_query = '''SELECT 
    state, city, COUNT(*) AS business_count
FROM
    yelp_business
GROUP BY state , city
ORDER BY business_count DESC
LIMIT {};'''.format(top_n)
cursor.execute(top_n_cities_query)
data = cursor.fetchall()
column_names = ["state","city","count"]
top_n_cities = pd.DataFrame(data,columns = column_names)


top_n_cities['count'] = top_n_cities['count'].apply(pd.to_numeric)
top_n_cities['city'] = top_n_cities['city'].astype(str)
top_n_cities['state'] = top_n_cities['state'].astype(str)


bubble_chart = BubbleChart(area=top_n_cities['count'], bubble_spacing=0.1)
bubble_chart.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
fig.set_size_inches(9, 13, forward=True)
bubble_chart.plot(ax, top_n_cities['state'],colors)
ax.axis("off")
ax.relim()
ax.autoscale_view()
st.pyplot(fig)
import folium
from streamlit_folium import folium_static 


#list of states and city with drop down for business map
all_state_query = "select distinct state from yelp_business"
cursor.execute(all_state_query)
data = cursor.fetchall()
column_names = ["state"]
all_state = pd.DataFrame(data,columns = column_names)
state_select = st.selectbox('Choose state', all_state)
if state_select != "":
    st.write('You selected ' + state_select)
    state_select =  "\'" +state_select + "\'" 
    state_city_query = "select distinct city from yelp_business where state = {}".format(state_select)
    cursor.execute(state_city_query)
    data = cursor.fetchall()
    column_names = ["city"]
    state_city = pd.DataFrame(data,columns = column_names)
    st.write('state city ' + state_city_query)
    city_select = st.selectbox('Choose city', state_city)
    st.write('city selected ' + city_select)
    city_select =  "\'" +city_select + "\'"
    if city_select != "" :
        city_caterories_query = '''
        SELECT C.categories
        FROM yelp_business as B INNER JOIN yelp_categories as C 
        ON B.business_id = C.business_id where B.city = {} limit 50;'''.format(city_select)
        cursor.execute(city_caterories_query)
        data = cursor.fetchall()
        column_names = ["categories"]
        city_caterories = pd.DataFrame(data,columns = column_names)
        st.write('caterogies ' + city_caterories)
        city_category_select = st.selectbox('Choose category', city_caterories)
        st.write('category selected ' + city_category_select)
        city_category_select =  "\'" +city_category_select + "\'"

        if city_category_select !="":
            get_category = '''
            SELECT B.name, B.latitude, B.longitude, C.categories,B.city,B.state,B.review_count
            FROM yelp_business as B INNER JOIN yelp_categories as C 
            ON B.business_id = C.business_id where B.city = {0} and C.categories = {1} ;
            '''.format(city_select,city_category_select)
            cursor.execute(get_category)
            data = cursor.fetchall()
            column_names = ["name","latitude","longitude","categories","city","state","review_count"]
            all_categories_df = pd.DataFrame(data,columns = column_names)
            st.write(all_categories_df)
            
            map_cat = folium.Map(location=[all_categories_df.latitude.mean(), all_categories_df.longitude.mean()], zoom_start=12,control_scale = True,prefer_canvas=True)
            folium.TileLayer('cartodbpositron').add_to(map_cat)
            for index, location_info in all_categories_df.iterrows():
                info = 'name: {}, review: {}'.format(location_info["name"].strip('\"'), location_info["review_count"])

                folium.Marker([location_info["latitude"], location_info["longitude"]], popup=info).add_to(map_cat)
            
            st.title('Map of' + city_category_select + 'in' +  city_select)
            folium_static(map_cat)
            
#             st.map(all_categories_df)


#             fig = px.scatter_mapbox(all_categories_df, lat="latitude", lon="longitude", hover_name="city", hover_data=["name","state", "review_count"],
#                                     zoom=10, height=300)
#             fig.update_layout(mapbox_style="carto-positron")
#             fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#             st.plotly_chart(fig, use_container_width=True)


