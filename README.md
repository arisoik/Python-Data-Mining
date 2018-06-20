This Project was implemented for academic purpose by <strong>Vasilis Kordalis</strong> and <strong>Aristeidis Oikonomou</strong>.

# Python-Data-Mining
<strong>Aim</strong> of this project is to manipulate some raw data. These data are some lat, lon, timestamp rows on a csv file which are part of bus trajectories.

<h2>Description</h2>
Given the train_set.csv (data_sets/train_set.csv) a new file containing routes is created names trips.csv (results/First_Group_of_Data/trips.csv)
After that this file is cleaned by corrupted data and a new file named tripsClean.csv is created (results/Clean_Routes/tripsClean.csv)

Afterwards there are a few things that are happening:

<li> Visualization of first 5 JourneyPatternId from tripsClean.csv</li>
<li> Find k-Nearest Neighbors for tripsClean.csv and test_set_a1.csv (data_sets/test_set_a1.csv)</li>
<li> Find 5 first-matching routes using LCS method for tripsClean.csv and test_set_a2.csv (data_sets/test_set_a2.csv)</li>
<br/>
Above are for playing with python. The main goal is Classification. For this some features are extraxted. These are:
<br/>
<br/>
<li> A Grid Sequence (the lat, long limits are extracted and a grid based on these is being calculated, using some width andd some hight. After that the points of the dataset are being mapped to the squares being drawn in the grid)</li>
<li> Start of bus route (the starting grid of the bus route)</li>
<li> End of bus route (the ending grid of the bus route)</li>
<li> Length of the route (the length of the grids of the bus route)</li>
<li> Grid axis route thickness (how thick in horizontal and vertical axis the route points are)</li>
