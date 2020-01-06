This Project was implemented for academic purpose by <strong>Vasilis Kordalis</strong> and <strong>Aristeidis Oikonomou</strong>.

# Python-Data-Mining
<strong>Aim</strong> of this project is to manipulate some bulk data. These data are lat-lon-timestamp rows on a csv file which are part of bus trajectories.

<h2>Description</h2>
We start with some data containing different bushes' trajectories file (data_sets/train_set.csv).
After a quick edit of that file, a new one containing the routes is created (results/First_Group_of_Data/trips.csv)
After that this file is cleaned from corrupted data and a new file is created (results/Clean_Routes/tripsClean.csv)

Afterwards there are a few things that are happening:

<li> Visualization of first 5 JourneyPatternId from tripsClean.csv</li>
<li> Find k-Nearest Neighbors of tripsClean.csv and write the results down to a new file (data_sets/test_set_a1.csv)</li>
<li> Find 5 first-matching routes using LCS method of tripsClean.csv and write the results down to a new file (data_sets/test_set_a2.csv)</li>
<br/>
After that classification comes. For this purpose some features are extraxted. These are:
<br/>
<br/>
<li> A Grid Sequence (the lat-lon max values are extracted and a grid based on these is being "designed" using a specific width and a specific hight for each cell. After that the points of the dataset are being mapped to the cells being "drawn" in the grid)</li>
<li> Start of bus route (the starting grid of the bus route)</li>
<li> End of bus route (the ending grid of the bus route)</li>
<li> Length of the route (the length of the grids of the bus route)</li>
<li> Grid axis route thickness (how thick in horizontal and vertical axis the route points are)</li>
