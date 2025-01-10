import pandas as pd
import folium

# Specify the path to your CSV file
file_path = 'Emergency Visits for Patients Residing in the LGA updated.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Clean the Longitude and Latitude columns
data['Longitude'] = data['Longitude'].astype(str).str.replace(' ', '').astype(float)
data['Latitude'] = data['Latitude'].astype(float)

# Define colors for each Triage Level
color_map = {
    'Semi Urgent (4)': 'red',
    'Non-Urgent (5)': 'blue',
    'Semi Urgent (4) & Non-Urgent (5) Combined': 'green'
}

# Create a base map centered around the average latitude and longitude
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=6)

# Add circle markers for each location in the DataFrame
for index, row in data.iterrows():
    triage_level = row['Triage Level']
    color = color_map.get(triage_level, 'gray')  # Default to gray if not found
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,  # Adjust the radius for size
        color=color,  # Circle color based on triage level
        fill=True,
        fill_color=color,  # Fill color
        fill_opacity=0.6,  # Opacity of the fill
        popup=row['LOCAL NAME']
    ).add_to(m)

legend_html = '''
    <div style="position: fixed; 
                bottom: 10px; left: 10px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);">
        <b>Legend</b><br>
        <i style="background:red; width: 15px; height: 15px; display: inline-block;"></i> Semi Urgent (4)<br>
        <i style="background:blue; width: 15px; height: 15px; display: inline-block;"></i> Non-Urgent (5)<br>
        <i style="background:green; width: 15px; height: 15px; display: inline-block;"></i> Semi Urgent (4) & Non-Urgent (5) Combined<br>
    </div>
'''

m.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
m.save('emergency_visits_map.html')

print("Map has been created and saved as 'emergency_visits_map.html'.")